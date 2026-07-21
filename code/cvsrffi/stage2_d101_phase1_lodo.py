"""Phase1-only nested receiver LODO diagnostics for the D101 alternative head.

The module reuses D99/D100's archive validator, episode builder, pseudo-new
rotation, base candidate evaluator, and metric implementation.  It never
reads target/r7 artifacts and can never mint target or formal Phase2 authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d99_d100_phase1_lodo as base
from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d100_ra_cgspr_lgf as d100
from cvsrffi import stage2_d101_shrinkage_rda as d101
from cvsrffi.stage2_d81_phase1_episode_scorer import D81Phase1EpisodeScorer


SCHEMA = "cvs.phase1.d101.nested_receiver_lodo.v1"
STATUS_ADMITTED = "PHASE1_DIAGNOSTIC_ADMITTED_PENDING_RESOURCE_AUTHORITY"
STATUS_REJECTED = "REJECTED_PHASE1_HARD_GATE"
ALLOWED_K = tuple(base.ALLOWED_K)
EPSILON = 1e-12
STRICT_TOLERANCE = 1e-12
ORACLE_GAIN_FLOOR = 0.0025
MAX_D101_GRID_CANDIDATES = 128

SHARED_D99_FIELDS = (
    "eta",
    "student_nu",
    "kernel_volume_gamma",
    "shared_h0",
    "scale_prior_strength",
    "scale_min_ratio",
    "scale_max_ratio",
    "d99_temperature",
)
D100_ONLY_FIELDS = ("lambda0", "ridge_temperature", "alpha")
D101_GRID_FIELDS = (
    "block_variance_z160",
    "block_variance_fft96",
    "block_variance_rf32",
    "prior_dof",
    "target_rank_k5plus",
    "lambda_relative",
    "rda_temperature",
    "d101_alpha",
)


class D101Phase1LODOError(ValueError):
    """Raised when Phase1-only selection, evidence, or receipt closure drifts."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def current_code_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "stage2_d101_phase1_lodo": _sha256_file(Path(__file__).resolve()),
        "stage2_d99_d100_phase1_lodo": _sha256_file(root / "stage2_d99_d100_phase1_lodo.py"),
        "stage2_d99_ra_cgtmk_d81": _sha256_file(root / "stage2_d99_ra_cgtmk_d81.py"),
        "stage2_d100_ra_cgspr_lgf": _sha256_file(root / "stage2_d100_ra_cgspr_lgf.py"),
        "stage2_d101_shrinkage_rda": _sha256_file(root / "stage2_d101_shrinkage_rda.py"),
        "stage2_d81_phase1_episode_scorer": _sha256_file(root / "stage2_d81_phase1_episode_scorer.py"),
    }


@dataclass(frozen=True, slots=True)
class D101LODOGateLock:
    minimum_top1_agreement: float = 0.98
    maximum_margin_sign_flip_rate: float = 0.005
    large_margin_threshold: float = 0.10
    maximum_large_margin_flip_count: int = 0
    oracle_union_gain_floor: float = ORACLE_GAIN_FLOOR
    strict_nll_tolerance: float = STRICT_TOLERANCE

    def __post_init__(self) -> None:
        values = (
            self.minimum_top1_agreement,
            self.maximum_margin_sign_flip_rate,
            self.large_margin_threshold,
            self.oracle_union_gain_floor,
            self.strict_nll_tolerance,
        )
        if (
            not all(math.isfinite(float(value)) for value in values)
            or not 0.0 <= self.minimum_top1_agreement <= 1.0
            or not 0.0 <= self.maximum_margin_sign_flip_rate <= 1.0
            or self.large_margin_threshold <= 0.0
            or self.maximum_large_margin_flip_count < 0
            or self.oracle_union_gain_floor < ORACLE_GAIN_FLOOR
            or self.strict_nll_tolerance < 0.0
        ):
            raise D101Phase1LODOError("invalid prelocked D101 LODO gate")

    @property
    def lock_digest(self) -> str:
        return canonical_sha256({"schema": f"{SCHEMA}.gate_lock", **asdict(self)})


def d101_candidate_grid(grid: Mapping[str, Iterable[float]]) -> list[dict[str, float]]:
    if set(grid) != set(D101_GRID_FIELDS):
        raise D101Phase1LODOError(
            f"D101 grid must have exact fields {list(D101_GRID_FIELDS)}"
        )
    columns: list[list[float]] = []
    for field in D101_GRID_FIELDS:
        try:
            values = sorted(float(item) for item in grid[field])
        except (TypeError, ValueError) as exc:
            raise D101Phase1LODOError(f"invalid D101 grid field {field}") from exc
        if not values or len(values) != len(set(values)) or not all(
            math.isfinite(item) for item in values
        ):
            raise D101Phase1LODOError(f"empty/nonfinite/duplicate D101 field {field}")
        columns.append(values)
    if math.prod(map(len, columns)) > MAX_D101_GRID_CANDIDATES:
        raise D101Phase1LODOError("D101 Cartesian grid exceeds 128 candidates")
    rows = [dict(zip(D101_GRID_FIELDS, values)) for values in itertools.product(*columns)]
    for row in rows:
        rank = row["target_rank_k5plus"]
        if (
            min(
                row["block_variance_z160"],
                row["block_variance_fft96"],
                row["block_variance_rf32"],
                row["prior_dof"],
                row["lambda_relative"],
                row["rda_temperature"],
            )
            <= 0.0
            or rank not in (0.0, 1.0, 2.0)
            or not 0.0 <= row["d101_alpha"] <= 1.0
        ):
            raise D101Phase1LODOError("D101 grid candidate violates frozen bounds")
    return rows


def _shared_signature(candidate: Mapping[str, float]) -> dict[str, float]:
    return {field: float(candidate[field]) for field in SHARED_D99_FIELDS}


def _normalize_folds(classes: Sequence[str]) -> tuple[dict[str, Any], ...]:
    raw = base.build_pseudo_new_folds(classes)
    normalized = []
    for fold in raw:
        pseudo_new = tuple(sorted(str(value) for value in fold["pseudo_new"]))
        pseudo_old = tuple(sorted(str(value) for value in fold["pseudo_old"]))
        normalized.append(
            {
                "fold_id": f"pseudo_new_{canonical_sha256(pseudo_new)[:16]}",
                "pseudo_new": pseudo_new,
                "pseudo_old": pseudo_old,
            }
        )
    return tuple(sorted(normalized, key=lambda item: item["pseudo_new"]))


def _positions(labels: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    return base._positions(np.asarray(labels).astype(str), classes)


def _incremental_metrics(
    after_probability: np.ndarray,
    before_old_probability: np.ndarray,
    after_truth: np.ndarray,
    before_old_truth: np.ndarray,
    classes: Sequence[str],
    pseudo_old: Sequence[str],
    pseudo_new: Sequence[str],
) -> dict[str, Any]:
    result = dict(
        base._metrics(after_probability, after_truth, classes, pseudo_old, pseudo_new)
    )
    before_prediction = np.argmax(before_old_probability, axis=1)
    b_old = float(
        np.mean(
            [
                np.mean(before_prediction[before_old_truth == index] == index)
                for index in range(len(pseudo_old))
            ]
        )
    )
    a_old = float(result["pseudo_old_accuracy"])
    result.update(
        {
            "B_old_pre_increment_accuracy": b_old,
            "A_old_post_increment_accuracy": a_old,
            "seen_new_accuracy": float(result["pseudo_new_accuracy"]),
            "H_old_new": float(result["harmonic_old_new"]),
            "all_registered_class_floor": float(result["worst_class_floor"]),
            "forgetting_B_minus_A": float(b_old - a_old),
            "before_state_is_independently_rebuilt_old_only": True,
            "before_state_is_not_post_head_logit_mask": True,
        }
    )
    return result


def _balanced_complementarity(
    left_probability: np.ndarray,
    right_probability: np.ndarray,
    truth: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    left = np.argmax(left_probability, axis=1)
    right = np.argmax(right_probability, axis=1)
    left_correct = left == truth
    right_correct = right == truth
    oracle = np.logical_or(left_correct, right_correct)
    per_class_oracle = [
        float(np.mean(oracle[truth == index])) for index in range(len(classes))
    ]
    return {
        "prediction_event_count": int(len(truth)),
        "disagreement_count": int(np.sum(left != right)),
        "right_correct_when_left_wrong_count": int(np.sum(right_correct & ~left_correct)),
        "left_correct_when_right_wrong_count": int(np.sum(left_correct & ~right_correct)),
        "class_balanced_oracle_union_accuracy": float(np.mean(per_class_oracle)),
        "same_episode_same_truth_same_class_balanced_denominator": True,
    }


def _pair_non_decreasing(
    d100_metrics: Mapping[str, Any],
    d101_metrics: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    higher = (
        "B_old_pre_increment_accuracy",
        "A_old_post_increment_accuracy",
        "seen_new_accuracy",
        "H_old_new",
        "all_registered_class_floor",
    )
    checks = {
        name: bool(float(d101_metrics[name]) >= float(d100_metrics[name]) - tolerance)
        for name in higher
    }
    checks["forgetting_not_increased"] = bool(
        float(d101_metrics["forgetting_B_minus_A"])
        <= float(d100_metrics["forgetting_B_minus_A"]) + tolerance
    )
    checks["balanced_nll_strictly_improved"] = bool(
        float(d101_metrics["balanced_nll"])
        < float(d100_metrics["balanced_nll"]) - tolerance
    )
    return {"passed": bool(all(checks.values())), "checks": checks}


def _candidate_d101_config(
    bank: d99.TypedINT8MetricKernelBank,
    ground: d99.GroundGeometry,
    candidate: Mapping[str, float],
    gate_lock: D101LODOGateLock,
    *,
    task_receipt_sha256: str,
) -> d101.Phase1D101Lock:
    rank = 0 if bank.metric.k_shot == 1 else int(candidate["target_rank_k5plus"])
    return d101.Phase1D101Lock(
        k_shot=int(bank.metric.k_shot),
        block_variance_prior=(
            float(candidate["block_variance_z160"]),
            float(candidate["block_variance_fft96"]),
            float(candidate["block_variance_rf32"]),
        ),
        prior_degrees_of_freedom=float(candidate["prior_dof"]),
        target_residual_rank=rank,
        lambda_relative=float(candidate["lambda_relative"]),
        temperature=float(candidate["rda_temperature"]),
        d99_temperature=float(candidate["d99_temperature"]),
        alpha=float(candidate["d101_alpha"]),
        d99_phase1_lock_digest=bank.config.lock_digest,
        ground_geometry_receipt_sha256=ground.geometry_receipt_sha256,
        phase1_lodo_receipt_sha256=canonical_sha256(
            [SCHEMA, gate_lock.lock_digest, task_receipt_sha256, candidate]
        ),
    )


def _rebuild_d101_teacher(
    bank: d99.TypedINT8MetricKernelBank,
    ground: d99.GroundGeometry,
    config: d101.Phase1D101Lock,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the FP64 analytic teacher transiently; never serialize it."""

    decoded = d99.decode_support_bank(bank)
    mapped = d101._feature_map(bank, decoded)
    indices = np.asarray(bank.class_indices_int16, dtype=np.int64)
    means, _residual, support_covariance = d101._within_class_covariance(
        mapped, indices, len(bank.classes), config.k_shot
    )
    empirical = d101._block_isotropic_variances(support_covariance)
    residual_dof = len(bank.classes) * (config.k_shot - 1)
    shrinkage = residual_dof / (config.prior_degrees_of_freedom + residual_dof)
    base_variance = (
        (1.0 - shrinkage) * np.asarray(config.block_variance_prior, dtype=np.float64)
        + shrinkage * empirical
    )
    target_values = np.empty(0, dtype=np.float64)
    target_vectors = np.empty((d101.FEATURE_DIM, 0), dtype=np.float64)
    if config.k_shot > 1 and config.target_residual_rank > 0:
        residual_covariance = support_covariance.copy()
        for block, variance in zip(d101.BLOCK_SLICES, empirical):
            residual_covariance[block, block] -= variance * np.eye(
                block.stop - block.start
            )
        values, vectors = np.linalg.eigh(
            0.5 * (residual_covariance + residual_covariance.T)
        )
        positive = np.flatnonzero(values > d101.EPSILON)
        rank = min(config.target_residual_rank, len(positive))
        if rank:
            order = positive[np.argsort(values[positive], kind="stable")[-rank:][::-1]]
            target_values = values[order]
            target_vectors = vectors[:, order]
    view = d101._shared_ground_covariance_view(ground)
    transformed_ground = d101._linear_d99_sqrt_on_ground_basis(bank, view)
    ground_weight = float(bank.metric.ground_weight)
    if ground_weight == 0.0 or transformed_ground.shape[1] == 0:
        ground_factor = np.zeros((d101.FEATURE_DIM, 0), dtype=np.float64)
    else:
        ground_factor = transformed_ground * np.sqrt(
            (1.0 - shrinkage)
            * ground_weight
            * view.nuisance_spectrum_fp32.astype(np.float64)
        )[None, :]
    target_factor = target_vectors * np.sqrt(shrinkage * target_values)[None, :]
    factor = np.concatenate((ground_factor, target_factor), axis=1)
    diagonal_bar = np.concatenate(
        [
            np.full(dim, variance, dtype=np.float64)
            for dim, variance in zip(d101.BLOCK_DIMS, base_variance)
        ]
    )
    trace = float(np.sum(diagonal_bar) + np.sum(np.square(factor)))
    diagonal = diagonal_bar + config.lambda_relative * trace / d101.FEATURE_DIM
    precision_means = d101._woodbury_precision_apply(means, diagonal, factor)
    bias = -0.5 * np.sum(means * precision_means, axis=1)
    return precision_means, bias


def held_quantization_margin_audit(
    state: d101.D101ShrinkageRDAState,
    bank: d99.TypedINT8MetricKernelBank,
    ground: d99.GroundGeometry,
    config: d101.Phase1D101Lock,
    query_features: np.ndarray,
    gate_lock: D101LODOGateLock,
) -> dict[str, Any]:
    teacher_weight, teacher_bias = _rebuild_d101_teacher(bank, ground, config)
    mapped = d101._feature_map(bank, np.ascontiguousarray(query_features, dtype=np.float32))
    teacher = mapped @ teacher_weight.T + teacher_bias[None, :]
    deployed = d101._score_compiled_rda_logits(
        state, bank, np.ascontiguousarray(query_features, dtype=np.float32)
    ).astype(np.float64)
    teacher_order = np.argsort(teacher, axis=1, kind="stable")
    rows = np.arange(len(teacher))
    winner = teacher_order[:, -1]
    runner_up = teacher_order[:, -2]
    margin = teacher[rows, winner] - teacher[rows, runner_up]
    deployed_teacher_margin = deployed[rows, winner] - deployed[rows, runner_up]
    flip = deployed_teacher_margin <= 0.0
    large = margin >= gate_lock.large_margin_threshold
    agreement = float(np.mean(np.argmax(teacher, axis=1) == np.argmax(deployed, axis=1)))
    flip_count = int(np.sum(flip))
    flip_rate = float(flip_count / max(len(teacher), 1))
    large_flip_count = int(np.sum(flip & large))
    checks = {
        "top1_agreement": bool(agreement >= gate_lock.minimum_top1_agreement),
        "margin_sign_flip_rate": bool(
            flip_rate <= gate_lock.maximum_margin_sign_flip_rate + EPSILON
        ),
        "large_margin_flip_count": bool(
            large_flip_count <= gate_lock.maximum_large_margin_flip_count
        ),
    }
    return {
        "scope": "held_phase1_evaluation_transient_fp64_teacher_vs_persistent_int8",
        "row_count": int(len(teacher)),
        "top1_agreement": agreement,
        "teacher_winner_margin_sign_flip_count": flip_count,
        "teacher_winner_margin_sign_flip_rate": flip_rate,
        "large_margin_threshold": float(gate_lock.large_margin_threshold),
        "large_margin_row_count": int(np.sum(large)),
        "large_margin_flip_count": large_flip_count,
        "teacher_persisted": False,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _typed_d81(
    logits: np.ndarray,
    features: np.ndarray,
    classes: Sequence[str],
    k_shot: int,
    scorer_contract: Mapping[str, Any],
) -> d100.TypedD81LogitBatch:
    return d100.bind_typed_d81_logits(
        np.ascontiguousarray(logits, dtype=np.float32),
        np.ascontiguousarray(features, dtype=np.float32),
        tuple(str(value) for value in classes),
        int(k_shot),
        source_schema=str(scorer_contract["schema"]),
        source_receipt_sha256=str(scorer_contract["receipt_sha256"]),
    )


def _old_only_probabilities(
    *,
    arrays: Mapping[str, np.ndarray],
    episode: base.Episode,
    query_indices: np.ndarray,
    pseudo_old: tuple[str, ...],
    local_bundle: d99.Phase1GroundAggregateBundle,
    base_d99_config: d99.Phase1D99Lock,
    base_candidate: Mapping[str, float],
    d101_candidate: Mapping[str, float],
    authority: base.GroundReleaseAuthority,
    old_d81_logits: np.ndarray,
    scorer_contract: Mapping[str, Any],
    gate_lock: D101LODOGateLock,
    task_receipt_sha256: str,
) -> tuple[dict[str, np.ndarray], d101.D101ShrinkageRDAState]:
    config99 = base._candidate_d99_config(
        base_d99_config,
        local_bundle,
        pseudo_old,
        base_candidate,
        episode.k_shot,
    )
    ground = d99.build_ground_geometry(local_bundle, config=config99)
    support_labels_all = arrays["labels"][episode.support].astype(str)
    keep = np.isin(support_labels_all, np.asarray(pseudo_old, dtype=str))
    support_indices = episode.support[keep]
    support_features = arrays["features"][support_indices]
    support_labels = arrays["labels"][support_indices].astype(str)
    support_physical = arrays["physical_ids"][support_indices].astype(str)
    metric = d99.fit_support_metric(
        ground,
        support_features,
        support_labels,
        support_physical,
        pseudo_old,
        pseudo_old,
        config=config99,
    )
    bank = d99.build_typed_support_bank(
        metric,
        support_features,
        support_labels,
        support_physical,
        pseudo_old,
        config=config99,
    )
    config100 = base._candidate_d100_config(config99, base_candidate, episode.k_shot, authority)
    state100 = d100.build_simplex_ridge_state(bank, config=config100)
    candidate101 = {**d101_candidate, "d99_temperature": base_candidate["d99_temperature"]}
    config101 = _candidate_d101_config(
        bank,
        ground,
        candidate101,
        gate_lock,
        task_receipt_sha256=task_receipt_sha256,
    )
    state101 = d101.build_shrinkage_rda_state(bank, ground, config=config101)
    old_mask = np.isin(
        arrays["labels"][query_indices].astype(str), np.asarray(pseudo_old, dtype=str)
    )
    old_query = np.ascontiguousarray(arrays["features"][query_indices[old_mask]], dtype=np.float32)
    typed = _typed_d81(old_d81_logits, old_query, pseudo_old, episode.k_shot, scorer_contract)
    fusion100 = d100.canonical_fuse_typed_d81_d99_d100(
        state100, bank, typed, old_query, evaluate_complementarity_branch=True
    )
    fusion101 = d101.canonical_fuse_typed_d81_d99_d101(
        state101, bank, typed, old_query, evaluate_complementarity_branch=True
    )
    return {
        "d81": fusion100.d81_probability_fp32,
        "d99": fusion100.d99_probability_fp32,
        "d100": fusion100.fused_probability_fp32,
        "d101": fusion101.fused_probability_fp32,
    }, state101


def _evaluate_joint_candidate(
    *,
    arrays: Mapping[str, np.ndarray],
    episode: base.Episode,
    query_indices: np.ndarray,
    fold: Mapping[str, Any],
    base_candidate: Mapping[str, float],
    d101_candidate: Mapping[str, float],
    base_d99_config: d99.Phase1D99Lock,
    outer_ground_bundle: d99.Phase1GroundAggregateBundle,
    authority: base.GroundReleaseAuthority,
    d81_logits: np.ndarray,
    old_d81_logits: np.ndarray,
    scorer_contract: Mapping[str, Any],
    gate_lock: D101LODOGateLock,
    prepared_cache: dict[tuple[Any, ...], tuple[Any, ...]],
    outer_held_receiver: str,
    split_name: str,
    d100_control_mode: str,
) -> dict[str, Any]:
    clean_candidate = {field: float(base_candidate[field]) for field in base._GRID_FIELDS}
    base_row = base._evaluate_candidate(
        arrays=arrays,
        episode=episode,
        query_indices=query_indices,
        fold=fold,
        candidate=clean_candidate,
        base_d99_config=base_d99_config,
        full_ground_bundle=outer_ground_bundle,
        authority=authority,
        d81_logits=d81_logits,
        d81_source_schema=str(scorer_contract["schema"]),
        d81_source_receipt_sha256=str(scorer_contract["receipt_sha256"]),
        prepared_cache=prepared_cache,
    )
    cache_key = (
        episode.receiver,
        episode.k_shot,
        fold["fold_id"],
        base.canonical_sha256(clean_candidate),
    )
    if cache_key not in prepared_cache:
        raise D101Phase1LODOError("base evaluator did not bind prepared cache")
    local_bundle, metric, bank, state100 = prepared_cache[cache_key]
    ground = d99.build_ground_geometry(local_bundle, config=bank.config)
    task_receipt = canonical_sha256(
        [
            outer_held_receiver,
            episode.receiver,
            episode.k_shot,
            fold["fold_id"],
            split_name,
            _array_receipt(query_indices),
            bank.bank_receipt_sha256,
        ]
    )
    candidate101 = {
        **{field: float(d101_candidate[field]) for field in D101_GRID_FIELDS},
        "d99_temperature": float(clean_candidate["d99_temperature"]),
    }
    config101 = _candidate_d101_config(
        bank,
        ground,
        candidate101,
        gate_lock,
        task_receipt_sha256=task_receipt,
    )
    state101 = d101.build_shrinkage_rda_state(bank, ground, config=config101)
    query_features = np.ascontiguousarray(arrays["features"][query_indices], dtype=np.float32)
    typed = _typed_d81(
        d81_logits, query_features, bank.classes, episode.k_shot, scorer_contract
    )
    fusion100 = d100.canonical_fuse_typed_d81_d99_d100(
        state100, bank, typed, query_features, evaluate_complementarity_branch=True
    )
    fusion101 = d101.canonical_fuse_typed_d81_d99_d101(
        state101, bank, typed, query_features, evaluate_complementarity_branch=True
    )
    if not np.array_equal(fusion100.d99_probability_fp32, fusion101.d99_probability_fp32):
        raise D101Phase1LODOError("D100/D101 did not share the exact p99 control")
    d100_fallback = d100_control_mode == "D99_FALLBACK_AFTER_D100_GUARD"
    if d100_fallback and (
        clean_candidate["alpha"] != 0.0
        or not np.array_equal(
            fusion100.fused_probability_fp32, fusion100.d99_probability_fp32
        )
    ):
        raise D101Phase1LODOError("D100 fallback must be exact effective-alpha-zero p99")
    pseudo_old = tuple(str(value) for value in fold["pseudo_old"])
    pseudo_new = tuple(str(value) for value in fold["pseudo_new"])
    before, _old_state101 = _old_only_probabilities(
        arrays=arrays,
        episode=episode,
        query_indices=query_indices,
        pseudo_old=pseudo_old,
        local_bundle=local_bundle,
        base_d99_config=base_d99_config,
        base_candidate=clean_candidate,
        d101_candidate=candidate101,
        authority=authority,
        old_d81_logits=old_d81_logits,
        scorer_contract=scorer_contract,
        gate_lock=gate_lock,
        task_receipt_sha256=task_receipt,
    )
    truth = _positions(arrays["labels"][query_indices], bank.classes)
    old_mask = np.isin(arrays["labels"][query_indices].astype(str), pseudo_old)
    old_truth = _positions(arrays["labels"][query_indices[old_mask]], pseudo_old)
    after_probability = {
        "d81": fusion100.d81_probability_fp32,
        "d99": fusion100.d99_probability_fp32,
        "d100": fusion100.fused_probability_fp32,
        "d101": fusion101.fused_probability_fp32,
    }
    metrics = {
        name: _incremental_metrics(
            after_probability[name],
            before[name],
            truth,
            old_truth,
            bank.classes,
            pseudo_old,
            pseudo_new,
        )
        for name in ("d81", "d99", "d100", "d101")
    }
    d99_d101 = _balanced_complementarity(
        after_probability["d99"], after_probability["d101"], truth, bank.classes
    )
    d99_d100 = _balanced_complementarity(
        after_probability["d99"], after_probability["d100"], truth, bank.classes
    )
    margin = held_quantization_margin_audit(
        state101, bank, ground, config101, query_features, gate_lock
    )
    episode_binding = {
        "outer_held_receiver": str(outer_held_receiver),
        "pseudo_target_receiver": str(episode.receiver),
        "k_shot": int(episode.k_shot),
        "fold_id": str(fold["fold_id"]),
        "split_name": str(split_name),
        "support_indices": episode.support.tolist(),
        "calibration_indices": episode.calibration.tolist(),
        "evaluation_indices": episode.evaluation.tolist(),
        "query_indices": np.asarray(query_indices, dtype=np.int64).tolist(),
        "support_receipt": _array_receipt(episode.support),
        "calibration_receipt": _array_receipt(episode.calibration),
        "evaluation_receipt": _array_receipt(episode.evaluation),
        "query_receipt": _array_receipt(np.asarray(query_indices, dtype=np.int64)),
    }
    if set(episode_binding["support_indices"]) & set(episode_binding["query_indices"]):
        raise D101Phase1LODOError("support/query split overlap")
    resource = dict(d101.audit_known_partial_combined_resources(state101, bank, ground))
    result = {
        "schema": f"{SCHEMA}.joint_row",
        "task_receipt_sha256": task_receipt,
        "outer_held_receiver": str(outer_held_receiver),
        "receiver": str(episode.receiver),
        "k_shot": int(episode.k_shot),
        "fold_id": str(fold["fold_id"]),
        "pseudo_old": list(pseudo_old),
        "pseudo_new": list(pseudo_new),
        "split_name": str(split_name),
        "base_candidate": clean_candidate,
        "base_candidate_sha256": canonical_sha256(clean_candidate),
        "d101_candidate": {field: candidate101[field] for field in D101_GRID_FIELDS},
        "d101_candidate_sha256": canonical_sha256(
            {field: candidate101[field] for field in D101_GRID_FIELDS}
        ),
        "episode_binding": episode_binding,
        "typed_d81_batch_receipt_sha256": typed.batch_receipt_sha256,
        "typed_d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "d100_state_receipt_sha256": state100.state_receipt_sha256,
        "d101_state_receipt_sha256": state101.state_receipt_sha256,
        "four_heads_same_episode_and_query_receipt": True,
        "d100_d101_share_exact_typed_d99_bank_and_p99": True,
        "d100_control_mode": str(d100_control_mode),
        "d100_effective_alpha": float(clean_candidate["alpha"]),
        "d100_fallback_prediction_exact_p99": bool(d100_fallback),
        "metrics": metrics,
        "d99_d101_complementarity": d99_d101,
        "d99_d100_complementarity": d99_d100,
        "d99_d101_changed_count": int(d99_d101["disagreement_count"]),
        "ground_coverage_rho": float(metric.ground_coverage_rho),
        "ground_weight": float(metric.ground_weight),
        "ground_bundle_receipt_sha256": local_bundle.bundle_sha256,
        "ground_domain_ids": list(local_bundle.domain_ids),
        "held_quantization_margin": margin,
        "support_fit_quantization_diagnostic": dict(state101.quantization_audit),
        "resource": resource,
        "resource_defer": "D81_PERSISTENT_HEAD_AND_COMPLETE_GROUND_WIRE_UNAVAILABLE",
        "formal_phase1_eligible": False,
        "target_authority": False,
    }
    result["joint_row_sha256"] = canonical_sha256(result)
    return result


def _aggregate_joint_rows(
    rows: Sequence[Mapping[str, Any]],
    gate_lock: D101LODOGateLock,
    *,
    k_shot: int,
) -> dict[str, Any]:
    if not rows:
        raise D101Phase1LODOError("cannot gate zero D101 joint rows")
    event_count = int(
        sum(row["d99_d101_complementarity"]["prediction_event_count"] for row in rows)
    )
    minimum_rescue = max(5, int(math.ceil(0.001 * event_count)))
    rescue_d101 = int(
        sum(
            row["d99_d101_complementarity"][
                "right_correct_when_left_wrong_count"
            ]
            for row in rows
        )
    )
    rescue_d99 = int(
        sum(
            row["d99_d101_complementarity"][
                "left_correct_when_right_wrong_count"
            ]
            for row in rows
        )
    )
    receiver_d101 = sorted(
        {
            str(row["receiver"])
            for row in rows
            if row["d99_d101_complementarity"][
                "right_correct_when_left_wrong_count"
            ]
            > 0
        }
    )
    receiver_d99 = sorted(
        {
            str(row["receiver"])
            for row in rows
            if row["d99_d101_complementarity"][
                "left_correct_when_right_wrong_count"
            ]
            > 0
        }
    )
    oracle101 = float(
        np.mean(
            [
                row["d99_d101_complementarity"][
                    "class_balanced_oracle_union_accuracy"
                ]
                for row in rows
            ]
        )
    )
    oracle100 = float(
        np.mean(
            [
                row["d99_d100_complementarity"][
                    "class_balanced_oracle_union_accuracy"
                ]
                for row in rows
            ]
        )
    )
    pair_gates = [
        {
            "receiver": row["receiver"],
            "outer_held_receiver": row["outer_held_receiver"],
            "fold_id": row["fold_id"],
            "pseudo_new": row["pseudo_new"],
            **_pair_non_decreasing(
                row["metrics"]["d100"],
                row["metrics"]["d101"],
                gate_lock.strict_nll_tolerance,
            ),
        }
        for row in rows
    ]
    k1_changed = int(sum(row["d99_d101_changed_count"] for row in rows))
    checks = {
        "d101_rescue_count": bool(rescue_d101 >= minimum_rescue),
        "d99_rescue_count": bool(rescue_d99 >= minimum_rescue),
        "d101_rescue_receiver_coverage": bool(len(receiver_d101) >= 2),
        "d99_rescue_receiver_coverage": bool(len(receiver_d99) >= 2),
        "oracle_union_gain": bool(
            oracle101 + EPSILON
            >= oracle100 + gate_lock.oracle_union_gain_floor
        ),
        "all_receiver_pseudo_new_pairs": bool(
            all(item["passed"] for item in pair_gates)
        ),
        "k1_nonidentity": bool(k_shot != 1 or k1_changed > 0),
        "held_quantization_margin": bool(
            all(row["held_quantization_margin"]["passed"] for row in rows)
        ),
    }
    mean_metrics = {
        head: {
            name: float(np.mean([row["metrics"][head][name] for row in rows]))
            for name in (
                "balanced_accuracy",
                "balanced_nll",
                "B_old_pre_increment_accuracy",
                "A_old_post_increment_accuracy",
                "seen_new_accuracy",
                "H_old_new",
                "forgetting_B_minus_A",
            )
        }
        for head in ("d81", "d99", "d100", "d101")
    }
    mean_metrics["d101"]["all_registered_class_floor_min"] = float(
        min(row["metrics"]["d101"]["all_registered_class_floor"] for row in rows)
    )
    return {
        "k_shot": int(k_shot),
        "prediction_event_definition": (
            "unique_(outer_receiver,pseudo_new_registration_task,query_index);"
            "same_physical_query_across_distinct_registration_tasks_is_a_distinct_event"
        ),
        "repeated_physical_query_across_distinct_registration_tasks": True,
        "prediction_event_count": event_count,
        "minimum_each_direction_rescue_count": minimum_rescue,
        "d101_correct_when_d99_wrong_count": rescue_d101,
        "d99_correct_when_d101_wrong_count": rescue_d99,
        "d101_rescue_receiver_set": receiver_d101,
        "d99_rescue_receiver_set": receiver_d99,
        "oracle_union_d99_d101_class_balanced": oracle101,
        "oracle_union_d99_d100_class_balanced": oracle100,
        "oracle_union_gain": float(oracle101 - oracle100),
        "oracle_union_gain_floor": float(gate_lock.oracle_union_gain_floor),
        "pair_gates": pair_gates,
        "k1_changed_count": k1_changed,
        "mean_metrics": mean_metrics,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _record_d99_candidate(
    candidate_id: str,
    shared_parameters: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    k_shot: int,
) -> dict[str, Any]:
    summary = base._aggregate(rows)
    guard = base.enforce_d99_guard(k_shot, summary)
    key = (
        0 if guard["d99_eligible"] else 1,
        float(summary["d99"]["balanced_nll"]),
        -float(summary["d99"]["worst_class_floor"]),
        str(candidate_id),
    )
    return {
        "candidate_id": str(candidate_id),
        "parameters": dict(shared_parameters),
        "rows": [_jsonable(row) for row in rows],
        "summary": summary,
        "guard": guard,
        "eligible": bool(guard["d99_eligible"]),
        "selection_key": list(key),
    }


def _record_d100_candidate(
    candidate_id: str,
    candidate: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = base._aggregate(rows)
    alpha_guard = base.enforce_alpha_guard(candidate, summary)
    eligible = bool(
        not alpha_guard["alpha_forced_zero"]
        and float(alpha_guard["effective_parameters"]["alpha"]) > 0.0
    )
    key = (
        0 if eligible else 1,
        float(summary["fused"]["balanced_nll"]),
        -float(summary["fused"]["worst_class_floor"]),
        str(candidate_id),
    )
    return {
        "candidate_id": str(candidate_id),
        "parameters": dict(candidate),
        "requested_parameters": dict(candidate),
        "guard_effective_parameters": dict(alpha_guard["effective_parameters"]),
        "rows": [_jsonable(row) for row in rows],
        "summary": summary,
        "alpha_guard": alpha_guard,
        "eligible": eligible,
        "selection_key": list(key),
    }


def _select_d100_control(
    records: Sequence[Mapping[str, Any]],
    selected_d99_parameters: Mapping[str, float],
) -> dict[str, Any] | None:
    if not records:
        return None
    positive = _winner(records)
    if positive is not None:
        effective = {
            field: float(positive["guard_effective_parameters"][field])
            for field in base._GRID_FIELDS
        }
        if effective["alpha"] <= 0.0:
            raise D101Phase1LODOError("positive D100 control lost positive alpha")
        return {
            "control_mode": "D100_POSITIVE_ALPHA",
            "source_requested_candidate_id": positive["candidate_id"],
            "effective_parameters": effective,
            "effective_control_id": canonical_sha256(effective),
            "fallback_to_d99_control": False,
        }
    source = min(records, key=lambda item: str(item["candidate_id"]))
    # lambda/ridge temperature cannot affect alpha-zero predictions.  Freeze
    # canonical data-independent placeholders so the effective control ID is
    # not spuriously changed by a failed requested D100 branch.
    effective = {
        **{field: float(selected_d99_parameters[field]) for field in SHARED_D99_FIELDS},
        "lambda0": 1.0,
        "ridge_temperature": 1.0,
        "alpha": 0.0,
    }
    return {
        "control_mode": "D99_FALLBACK_AFTER_D100_GUARD",
        "source_requested_candidate_id": source["candidate_id"],
        "effective_parameters": effective,
        "effective_control_id": canonical_sha256(effective),
        "fallback_to_d99_control": True,
    }


def _record_d101_candidate(
    candidate_id: str,
    candidate: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    gate_lock: D101LODOGateLock,
    k_shot: int,
) -> dict[str, Any]:
    gate = _aggregate_joint_rows(rows, gate_lock, k_shot=k_shot)
    key = (
        0 if gate["passed"] else 1,
        float(gate["mean_metrics"]["d101"]["balanced_nll"]),
        -float(gate["mean_metrics"]["d101"]["all_registered_class_floor_min"]),
        str(candidate_id),
    )
    return {
        "candidate_id": str(candidate_id),
        "parameters": dict(candidate),
        "rows": [_jsonable(row) for row in rows],
        "gate": gate,
        "eligible": bool(gate["passed"]),
        "selection_key": list(key),
    }


def _winner(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [record for record in records if record["eligible"]]
    return min(eligible, key=lambda item: tuple(item["selection_key"])) if eligible else None


def _validate_run_inputs(
    archive_path: str | Path,
    archive_manifest_path: str | Path,
    archive_manifest_sha256: str,
    *,
    ground_bundle: d99.Phase1GroundAggregateBundle,
    ground_authority: base.GroundReleaseAuthority,
    base_d99_config: d99.Phase1D99Lock,
    base_scorer: D81Phase1EpisodeScorer,
    base_scorer_id: str,
    base_scorer_receipt_sha256: str,
    code_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(ground_bundle) is not d99.Phase1GroundAggregateBundle:
        raise D101Phase1LODOError("exact typed Phase1 ground bundle required")
    if type(ground_authority) is not base.GroundReleaseAuthority:
        raise D101Phase1LODOError("exact loaded ground authority required")
    if ground_authority.loader_token is not base._GROUND_AUTHORITY_TOKEN:
        raise D101Phase1LODOError("ground authority loader token drift")
    if (
        ground_authority.bundle_sha256 != ground_bundle.bundle_sha256
        or ground_authority.aggregation_receipt_sha256
        != ground_bundle.aggregation_receipt.receipt_sha256
        or tuple(base_d99_config.ground_old_registry)
        != tuple(ground_bundle.ground_old_registry)
    ):
        raise D101Phase1LODOError("ground/base D99 closure drift")
    normalized_code = {str(key): str(value).lower() for key, value in code_sha256.items()}
    if normalized_code != current_code_sha256():
        raise D101Phase1LODOError("D101 LODO code SHA registry/source drift")
    scorer_contract = base._validate_base_scorer_contract(
        base_scorer,
        expected_scorer_id=base_scorer_id,
        expected_receipt_sha256=base_scorer_receipt_sha256,
    )
    validated = base.validate_feature_archive(archive_path)
    manifest = base._validate_feature_archive_manifest(
        archive_manifest_path,
        archive_manifest_sha256,
        validated=validated,
    )
    arrays = validated["arrays"]
    classes = tuple(str(value) for value in arrays["class_ids"].tolist())
    receivers = tuple(sorted(validated["receivers"].astype(str).tolist()))
    if (
        set(classes) != set(ground_bundle.ground_old_registry)
        or set(receivers) != set(ground_authority.receiver_domain_map)
        or set(ground_authority.receiver_domain_map.values()) != set(ground_bundle.domain_ids)
    ):
        raise D101Phase1LODOError("archive receiver/class registry and ground authority differ")
    return validated, manifest, scorer_contract


def _precompute_d81_logits(
    arrays: Mapping[str, np.ndarray],
    episodes: Mapping[str, Mapping[int, base.Episode]],
    folds: Sequence[Mapping[str, Any]],
    classes: tuple[str, ...],
    receivers: tuple[str, ...],
    scorer: D81Phase1EpisodeScorer,
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[tuple[Any, ...], np.ndarray]]:
    all_class: dict[tuple[Any, ...], np.ndarray] = {}
    old_only: dict[tuple[Any, ...], np.ndarray] = {}
    for receiver in receivers:
        for k_shot in ALLOWED_K:
            episode = episodes[receiver][k_shot]
            for split_name, indices in (
                ("calibration", episode.calibration),
                ("evaluation", episode.evaluation),
            ):
                all_class[(receiver, k_shot, split_name)] = np.ascontiguousarray(
                    scorer(
                        arrays["features"][episode.support],
                        arrays["labels"][episode.support].astype(str),
                        arrays["features"][indices],
                        np.asarray(classes),
                    ),
                    dtype=np.float32,
                )
            support_labels = arrays["labels"][episode.support].astype(str)
            for fold in folds:
                pseudo_old = tuple(fold["pseudo_old"])
                support_keep = np.isin(support_labels, pseudo_old)
                old_support = episode.support[support_keep]
                for split_name, indices in (
                    ("calibration", episode.calibration),
                    ("evaluation", episode.evaluation),
                ):
                    old_query = indices[
                        np.isin(arrays["labels"][indices].astype(str), pseudo_old)
                    ]
                    old_only[(receiver, k_shot, fold["fold_id"], split_name)] = (
                        np.ascontiguousarray(
                            scorer(
                                arrays["features"][old_support],
                                arrays["labels"][old_support].astype(str),
                                arrays["features"][old_query],
                                np.asarray(pseudo_old),
                            ),
                            dtype=np.float32,
                        )
                    )
    return all_class, old_only


def _evaluate_base_calibration(
    *,
    arrays: Mapping[str, np.ndarray],
    episode: base.Episode,
    fold: Mapping[str, Any],
    candidate: Mapping[str, float],
    base_d99_config: d99.Phase1D99Lock,
    outer_bundle: d99.Phase1GroundAggregateBundle,
    authority: base.GroundReleaseAuthority,
    d81_logits: np.ndarray,
    scorer_contract: Mapping[str, Any],
    prepared_cache: dict[tuple[Any, ...], tuple[Any, ...]],
    outer_receiver: str,
    train_receivers: Sequence[str],
) -> dict[str, Any]:
    evaluated = base._evaluate_candidate(
        arrays=arrays,
        episode=episode,
        query_indices=episode.calibration,
        fold=fold,
        candidate=candidate,
        base_d99_config=base_d99_config,
        full_ground_bundle=outer_bundle,
        authority=authority,
        d81_logits=d81_logits,
        d81_source_schema=str(scorer_contract["schema"]),
        d81_source_receipt_sha256=str(scorer_contract["receipt_sha256"]),
        prepared_cache=prepared_cache,
    )
    cache_key = (
        episode.receiver,
        episode.k_shot,
        fold["fold_id"],
        base.canonical_sha256(candidate),
    )
    local_bundle, _metric, bank, _state100 = prepared_cache[cache_key]
    result = dict(evaluated)
    result["d101_selection_binding"] = {
        "outer_held_receiver": outer_receiver,
        "outer_train_receivers": list(train_receivers),
        "support_indices": episode.support.tolist(),
        "query_indices": episode.calibration.tolist(),
        "support_receipt": _array_receipt(episode.support),
        "query_receipt": _array_receipt(episode.calibration),
        "typed_d99_bank_receipt_sha256": bank.bank_receipt_sha256,
        "ground_bundle_receipt_sha256": local_bundle.bundle_sha256,
        "ground_domain_ids": list(local_bundle.domain_ids),
    }
    return result


def run_phase1_d101_nested_lodo(
    archive_path: str | Path,
    archive_manifest_path: str | Path,
    archive_manifest_sha256: str,
    *,
    ground_bundle: d99.Phase1GroundAggregateBundle,
    ground_authority: base.GroundReleaseAuthority,
    base_d99_config: d99.Phase1D99Lock,
    base_scorer: D81Phase1EpisodeScorer,
    base_scorer_id: str,
    base_scorer_receipt_sha256: str,
    d99_d100_grid: Mapping[str, Iterable[float]],
    d101_grid: Mapping[str, Iterable[float]],
    gate_lock: D101LODOGateLock,
    code_sha256: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    """Run pure Phase1 nested receiver LODO; never mint target authority."""

    if type(gate_lock) is not D101LODOGateLock:
        raise D101Phase1LODOError("exact D101 gate lock required")
    validated, manifest, scorer_contract = _validate_run_inputs(
        archive_path,
        archive_manifest_path,
        archive_manifest_sha256,
        ground_bundle=ground_bundle,
        ground_authority=ground_authority,
        base_d99_config=base_d99_config,
        base_scorer=base_scorer,
        base_scorer_id=base_scorer_id,
        base_scorer_receipt_sha256=base_scorer_receipt_sha256,
        code_sha256=code_sha256,
    )
    arrays = validated["arrays"]
    classes = tuple(str(value) for value in arrays["class_ids"].tolist())
    receivers = tuple(sorted(validated["receivers"].astype(str).tolist()))
    episodes = base.build_receiver_lodo_episodes(validated, seed=int(seed))
    folds = _normalize_folds(classes)
    base_candidates = base.candidate_grid(d99_d100_grid)
    candidates101 = d101_candidate_grid(d101_grid)
    all_logits, old_logits = _precompute_d81_logits(
        arrays, episodes, folds, classes, receivers, base_scorer
    )
    base._recheck_base_scorer_contract(base_scorer, scorer_contract)

    groups: dict[str, list[tuple[int, dict[str, float]]]] = {}
    for index, candidate in enumerate(base_candidates):
        signature = _shared_signature(candidate)
        groups.setdefault(canonical_sha256(signature), []).append((index, candidate))

    k_results: dict[str, Any] = {}
    all_outer_rows: list[dict[str, Any]] = []
    for k_shot in ALLOWED_K:
        outer_selections: list[dict[str, Any]] = []
        selected_outer_rows: list[dict[str, Any]] = []
        for outer_receiver in receivers:
            outer_domain = ground_authority.receiver_domain_map[outer_receiver]
            outer_bundle = base._subset_ground_bundle(
                ground_bundle, held_domain=outer_domain, pseudo_old=classes
            )
            if outer_domain in outer_bundle.domain_ids:
                raise D101Phase1LODOError("outer held ground domain was not removed")
            train_receivers = tuple(value for value in receivers if value != outer_receiver)
            prepared_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
            base_rows: dict[int, list[dict[str, Any]]] = {}
            for candidate_index, candidate in enumerate(base_candidates):
                rows = []
                for receiver in train_receivers:
                    episode = episodes[receiver][k_shot]
                    for fold in folds:
                        rows.append(
                            _evaluate_base_calibration(
                                arrays=arrays,
                                episode=episode,
                                fold=fold,
                                candidate=candidate,
                                base_d99_config=base_d99_config,
                                outer_bundle=outer_bundle,
                                authority=ground_authority,
                                d81_logits=all_logits[
                                    (receiver, k_shot, "calibration")
                                ],
                                scorer_contract=scorer_contract,
                                prepared_cache=prepared_cache,
                                outer_receiver=outer_receiver,
                                train_receivers=train_receivers,
                            )
                        )
                base_rows[candidate_index] = rows

            d99_records = []
            for signature_id, members in sorted(groups.items()):
                representative_index, representative = members[0]
                d99_records.append(
                    _record_d99_candidate(
                        signature_id,
                        _shared_signature(representative),
                        base_rows[representative_index],
                        k_shot,
                    )
                )
            selected_d99 = _winner(d99_records)
            selection: dict[str, Any] = {
                "outer_held_receiver": outer_receiver,
                "outer_train_receivers": list(train_receivers),
                "outer_held_ground_domain": outer_domain,
                "outer_ground_domain_ids": list(outer_bundle.domain_ids),
                "prepared_cache_scope": f"outer_only:{outer_receiver}",
                "d99_candidates": d99_records,
                "d100_candidates": [],
                "d101_candidates": [],
                "selected_d99_candidate_id": None,
                "selected_d100_candidate_id": None,
                "selected_d100_source_requested_candidate_id": None,
                "selected_d100_control": None,
                "selected_d101_candidate_id": None,
                "selection_status": "REJECTED_D99",
            }
            if selected_d99 is None:
                outer_selections.append(selection)
                continue
            selected_signature_id = str(selected_d99["candidate_id"])
            selection["selected_d99_candidate_id"] = selected_signature_id
            member_candidates = groups[selected_signature_id]
            d100_records = [
                _record_d100_candidate(
                    canonical_sha256(candidate), candidate, base_rows[index]
                )
                for index, candidate in member_candidates
            ]
            selection["d100_candidates"] = d100_records
            selected_d100_control = _select_d100_control(
                d100_records, selected_d99["parameters"]
            )
            if selected_d100_control is None:
                raise AssertionError("nonempty D100 candidates must yield a control")
            selected_base_candidate = dict(
                selected_d100_control["effective_parameters"]
            )
            selection["selected_d100_candidate_id"] = selected_d100_control[
                "effective_control_id"
            ]
            selection["selected_d100_source_requested_candidate_id"] = (
                selected_d100_control["source_requested_candidate_id"]
            )
            selection["selected_d100_control"] = selected_d100_control

            d101_records = []
            for candidate101 in candidates101:
                rows101 = []
                for receiver in train_receivers:
                    episode = episodes[receiver][k_shot]
                    for fold in folds:
                        rows101.append(
                            _evaluate_joint_candidate(
                                arrays=arrays,
                                episode=episode,
                                query_indices=episode.calibration,
                                fold=fold,
                                base_candidate=selected_base_candidate,
                                d101_candidate=candidate101,
                                base_d99_config=base_d99_config,
                                outer_ground_bundle=outer_bundle,
                                authority=ground_authority,
                                d81_logits=all_logits[
                                    (receiver, k_shot, "calibration")
                                ],
                                old_d81_logits=old_logits[
                                    (receiver, k_shot, fold["fold_id"], "calibration")
                                ],
                                scorer_contract=scorer_contract,
                                gate_lock=gate_lock,
                                prepared_cache=prepared_cache,
                                outer_held_receiver=outer_receiver,
                                split_name="inner_calibration",
                                d100_control_mode=selected_d100_control["control_mode"],
                            )
                        )
                candidate_id = canonical_sha256(candidate101)
                d101_records.append(
                    _record_d101_candidate(
                        candidate_id, candidate101, rows101, gate_lock, k_shot
                    )
                )
            selection["d101_candidates"] = d101_records
            selected_d101 = _winner(d101_records)
            if selected_d101 is None:
                selection["selection_status"] = "REJECTED_D101_INNER_GATE"
                outer_selections.append(selection)
                continue
            selection["selected_d101_candidate_id"] = selected_d101["candidate_id"]
            selection["selection_status"] = "NESTED_WINNERS_FROZEN"
            outer_selections.append(selection)
            selected_candidate101 = dict(selected_d101["parameters"])
            outer_episode = episodes[outer_receiver][k_shot]
            for fold in folds:
                selected_outer_rows.append(
                    _evaluate_joint_candidate(
                        arrays=arrays,
                        episode=outer_episode,
                        query_indices=outer_episode.evaluation,
                        fold=fold,
                        base_candidate=selected_base_candidate,
                        d101_candidate=selected_candidate101,
                        base_d99_config=base_d99_config,
                        outer_ground_bundle=outer_bundle,
                        authority=ground_authority,
                        d81_logits=all_logits[
                            (outer_receiver, k_shot, "evaluation")
                        ],
                        old_d81_logits=old_logits[
                            (outer_receiver, k_shot, fold["fold_id"], "evaluation")
                        ],
                        scorer_contract=scorer_contract,
                        gate_lock=gate_lock,
                        prepared_cache=prepared_cache,
                        outer_held_receiver=outer_receiver,
                        split_name="outer_held_evaluation",
                        d100_control_mode=selected_d100_control["control_mode"],
                    )
                )
        all_outer_selected = bool(
            len(selected_outer_rows) == len(receivers) * len(folds)
            and all(
                item["selection_status"] == "NESTED_WINNERS_FROZEN"
                for item in outer_selections
            )
        )
        outer_gate = (
            _aggregate_joint_rows(selected_outer_rows, gate_lock, k_shot=k_shot)
            if all_outer_selected
            else None
        )
        k_results[str(k_shot)] = {
            "outer_selections": outer_selections,
            "outer_held_evaluation_rows": selected_outer_rows,
            "all_outer_nested_winners_available": all_outer_selected,
            "outer_hard_gate": outer_gate,
            "passed": bool(all_outer_selected and outer_gate and outer_gate["passed"]),
        }
        all_outer_rows.extend(selected_outer_rows)

    scientific_pass = bool(all(k_results[str(k)]["passed"] for k in ALLOWED_K))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS_ADMITTED if scientific_pass else STATUS_REJECTED,
        "scientific_phase1_hard_gate_passed": scientific_pass,
        "formal_phase1_lock": False,
        "formal_phase2_eligible": False,
        "target_authority": False,
        "n607_authority": False,
        "canonical_lock_artifact_write_allowed": False,
        "seed": int(seed),
        "classes_canonical_set": sorted(classes),
        "receivers_canonical": list(receivers),
        "folds_canonical": list(folds),
        "gate_lock": asdict(gate_lock),
        "gate_lock_digest": gate_lock.lock_digest,
        "base_candidates": base_candidates,
        "base_grid_digest": canonical_sha256(base_candidates),
        "d101_candidates": candidates101,
        "d101_grid_digest": canonical_sha256(candidates101),
        "code_sha256": dict(code_sha256),
        "archive": {
            "manifest": manifest,
            "array_archive_sha256": validated["archive_sha256"],
        },
        "ground": {
            "bundle_sha256": ground_bundle.bundle_sha256,
            "authority_manifest_sha256": ground_authority.manifest_sha256,
            "receiver_domain_map": dict(ground_authority.receiver_domain_map),
            "domain_ids": list(ground_bundle.domain_ids),
        },
        "k_results": k_results,
        "protocol_audit": {
            "phase1_only": True,
            "target_or_r7_rows_used": 0,
            "r7_final_metrics_used_for_d101_selection": False,
            "query_truth_used_only_for_phase1_held_evaluation_metrics": True,
            "outer_held_receiver_excluded_from_inner_selection": True,
            "outer_held_ground_domain_removed_before_inner_and_outer_evaluation": True,
            "prepared_cache_isolated_per_outer_receiver": True,
            "before_four_heads_use_old_only_registry_and_support": True,
            "post_head_mask_not_used_as_before_state": True,
            "d100_and_d101_nested_winners_selected_independently": True,
            "fixed_upstream_d81_or_encoder_may_include_held_receiver": True,
            "whole_method_unseen_receiver_generalization_claim": False,
            "only_aggregated_ground_knowledge_head_lodo_claim": True,
        },
        "quantization_audit": {
            "held_teacher_rebuilt_transiently": True,
            "held_margin_row_count": int(
                sum(
                    row["held_quantization_margin"]["row_count"]
                    for row in all_outer_rows
                )
            ),
            "teacher_persisted": False,
        },
        "resource_audit": {
            "scope": "KNOWN_PARTIAL_EXACT_OBJECT_SERIALIZERS_ONLY",
            "selected_outer_row_count": len(all_outer_rows),
            "complete_d81_persistent_head_wire_available": False,
            "complete_ground_wire_available": False,
            "complete_combined_resource_claim": False,
            "formal_under_256kib_claim": False,
            "resource_defer": "D81_PERSISTENT_HEAD_AND_COMPLETE_GROUND_WIRE_UNAVAILABLE",
        },
        "deferred_authorities": [
            "complete_D81_persistent_head_resource",
            "complete_ground_wire_resource",
            "independent_formal_release_authority",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    if not verify_receipt(payload):
        raise D101Phase1LODOError("constructed D101 LODO receipt failed semantic closure")
    return payload


def _rebuild_grid_from_candidates(
    candidates: Sequence[Mapping[str, Any]], fields: Sequence[str], builder: Any
) -> list[dict[str, float]]:
    grid = {
        field: sorted({float(candidate[field]) for candidate in candidates})
        for field in fields
    }
    return builder(grid)


def _verify_base_selection_row(
    row: Mapping[str, Any], outer_receiver: str, outer_domain: str
) -> bool:
    binding = row.get("d101_selection_binding", {})
    support = set(int(value) for value in binding.get("support_indices", []))
    query = set(int(value) for value in binding.get("query_indices", []))
    return bool(
        binding.get("outer_held_receiver") == outer_receiver
        and outer_receiver not in set(binding.get("outer_train_receivers", []))
        and not support.intersection(query)
        and outer_domain not in set(binding.get("ground_domain_ids", []))
        and binding.get("support_receipt")
        == _array_receipt(np.asarray(binding.get("support_indices", []), dtype=np.int64))
        and binding.get("query_receipt")
        == _array_receipt(np.asarray(binding.get("query_indices", []), dtype=np.int64))
        and isinstance(binding.get("typed_d99_bank_receipt_sha256"), str)
        and len(binding["typed_d99_bank_receipt_sha256"]) == 64
    )


def _verify_joint_row(
    row: Mapping[str, Any],
    outer_receiver: str,
    outer_domain: str,
    gate_lock: D101LODOGateLock | None = None,
) -> bool:
    core = dict(row)
    receipt = core.pop("joint_row_sha256", None)
    if receipt != canonical_sha256(core):
        return False
    binding = row.get("episode_binding", {})
    support = set(int(value) for value in binding.get("support_indices", []))
    calibration = set(int(value) for value in binding.get("calibration_indices", []))
    evaluation = set(int(value) for value in binding.get("evaluation_indices", []))
    query = set(int(value) for value in binding.get("query_indices", []))
    expected_query = (
        evaluation
        if binding.get("split_name") == "outer_held_evaluation"
        else calibration
    )
    before_flags = all(
        row.get("metrics", {}).get(head, {}).get(
            "before_state_is_independently_rebuilt_old_only"
        )
        is True
        and row.get("metrics", {}).get(head, {}).get(
            "before_state_is_not_post_head_logit_mask"
        )
        is True
        for head in ("d81", "d99", "d100", "d101")
    )
    control_mode = row.get("d100_control_mode")
    effective_alpha = float(row.get("d100_effective_alpha", math.nan))
    control_semantic = bool(
        (
            control_mode == "D99_FALLBACK_AFTER_D100_GUARD"
            and effective_alpha == 0.0
            and row.get("d100_fallback_prediction_exact_p99") is True
        )
        or (
            control_mode == "D100_POSITIVE_ALPHA"
            and effective_alpha > 0.0
            and row.get("d100_fallback_prediction_exact_p99") is False
        )
    )
    margin = row.get("held_quantization_margin", {})
    margin_semantic = True
    if gate_lock is not None:
        row_count = int(margin.get("row_count", -1))
        flip_count = int(margin.get("teacher_winner_margin_sign_flip_count", -1))
        flip_rate = float(margin.get("teacher_winner_margin_sign_flip_rate", math.nan))
        large_flip = int(margin.get("large_margin_flip_count", -1))
        expected_checks = {
            "top1_agreement": bool(
                float(margin.get("top1_agreement", math.nan))
                >= gate_lock.minimum_top1_agreement
            ),
            "margin_sign_flip_rate": bool(
                flip_rate <= gate_lock.maximum_margin_sign_flip_rate + EPSILON
            ),
            "large_margin_flip_count": bool(
                large_flip <= gate_lock.maximum_large_margin_flip_count
            ),
        }
        margin_semantic = bool(
            row_count > 0
            and 0 <= flip_count <= row_count
            and math.isclose(flip_rate, flip_count / row_count, abs_tol=EPSILON)
            and margin.get("large_margin_threshold")
            == gate_lock.large_margin_threshold
            and margin.get("checks") == expected_checks
            and margin.get("passed") is bool(all(expected_checks.values()))
        )
    return bool(
        row.get("outer_held_receiver") == outer_receiver
        and binding.get("outer_held_receiver") == outer_receiver
        and not support.intersection(calibration)
        and not support.intersection(evaluation)
        and not calibration.intersection(evaluation)
        and query == expected_query
        and outer_domain not in set(row.get("ground_domain_ids", []))
        and row.get("four_heads_same_episode_and_query_receipt") is True
        and row.get("d100_d101_share_exact_typed_d99_bank_and_p99") is True
        and before_flags
        and control_semantic
        and margin_semantic
        and row.get("formal_phase1_eligible") is False
        and row.get("target_authority") is False
        and row.get("held_quantization_margin", {}).get("teacher_persisted") is False
        and row.get("resource", {}).get("complete_combined_resource_claim") is False
    )


def verify_receipt(value: Mapping[str, Any]) -> bool:
    """Verify both canonical bytes and selection/gate semantics."""

    try:
        receipt = _jsonable(value)
        stored_sha = receipt.pop("receipt_sha256")
        if stored_sha != canonical_sha256(receipt) or receipt["schema"] != SCHEMA:
            return False
        if (
            receipt["formal_phase1_lock"] is not False
            or receipt["formal_phase2_eligible"] is not False
            or receipt["target_authority"] is not False
            or receipt["n607_authority"] is not False
            or receipt["canonical_lock_artifact_write_allowed"] is not False
            or receipt["protocol_audit"]["whole_method_unseen_receiver_generalization_claim"]
            is not False
            or receipt["protocol_audit"]["fixed_upstream_d81_or_encoder_may_include_held_receiver"]
            is not True
            or receipt["protocol_audit"]["r7_final_metrics_used_for_d101_selection"]
            is not False
            or receipt["protocol_audit"]["target_or_r7_rows_used"] != 0
            or receipt["resource_audit"]["complete_combined_resource_claim"] is not False
            or receipt["resource_audit"]["formal_under_256kib_claim"] is not False
        ):
            return False
        if receipt["code_sha256"] != current_code_sha256():
            return False
        gate_lock = D101LODOGateLock(**receipt["gate_lock"])
        if gate_lock.lock_digest != receipt["gate_lock_digest"]:
            return False
        rebuilt_base = _rebuild_grid_from_candidates(
            receipt["base_candidates"], base._GRID_FIELDS, base.candidate_grid
        )
        rebuilt101 = _rebuild_grid_from_candidates(
            receipt["d101_candidates"], D101_GRID_FIELDS, d101_candidate_grid
        )
        if (
            rebuilt_base != receipt["base_candidates"]
            or rebuilt101 != receipt["d101_candidates"]
            or canonical_sha256(rebuilt_base) != receipt["base_grid_digest"]
            or canonical_sha256(rebuilt101) != receipt["d101_grid_digest"]
        ):
            return False
        receivers = tuple(receipt["receivers_canonical"])
        if tuple(sorted(receivers)) != receivers or len(set(receivers)) != len(receivers):
            return False
        folds = receipt["folds_canonical"]
        if folds != _jsonable(_normalize_folds(receipt["classes_canonical_set"])):
            return False
        receiver_domain_map = receipt["ground"]["receiver_domain_map"]
        all_ground_domains = tuple(receipt["ground"]["domain_ids"])
        if set(all_ground_domains) != set(receiver_domain_map.values()):
            return False
        fold_lookup = {fold["fold_id"]: fold for fold in folds}
        fold_ids = tuple(sorted(fold_lookup))
        expected_d99_candidate_ids = sorted(
            {
                canonical_sha256(_shared_signature(candidate))
                for candidate in receipt["base_candidates"]
            }
        )
        expected_d100_candidate_ids_by_d99 = {
            signature_id: [
                canonical_sha256(candidate)
                for candidate in receipt["base_candidates"]
                if canonical_sha256(_shared_signature(candidate)) == signature_id
            ]
            for signature_id in expected_d99_candidate_ids
        }
        expected_d101_candidate_ids = [
            canonical_sha256(candidate) for candidate in receipt["d101_candidates"]
        ]
        scientific_passes = []
        for k_shot in ALLOWED_K:
            result = receipt["k_results"][str(k_shot)]
            verified_outer_rows = []
            all_selected = True
            selection_receivers = [
                selection["outer_held_receiver"]
                for selection in result["outer_selections"]
            ]
            stored_outer_rows = result["outer_held_evaluation_rows"]
            stored_outer_keys = [
                (row["outer_held_receiver"], row["fold_id"])
                for row in stored_outer_rows
            ]
            expected_outer_keys = {
                (receiver, fold_id) for receiver in receivers for fold_id in fold_ids
            }
            expected_stored_outer_keys = {
                (selection["outer_held_receiver"], fold_id)
                for selection in result["outer_selections"]
                if selection["selection_status"] == "NESTED_WINNERS_FROZEN"
                for fold_id in fold_ids
            }
            if (
                selection_receivers != list(receivers)
                or len(stored_outer_keys) != len(set(stored_outer_keys))
                or not set(stored_outer_keys).issubset(expected_outer_keys)
                or set(stored_outer_keys) != expected_stored_outer_keys
            ):
                return False
            for selection in result["outer_selections"]:
                outer = selection["outer_held_receiver"]
                outer_domain = receiver_domain_map[outer]
                train_receivers = tuple(value for value in receivers if value != outer)
                expected_inner_keys = {
                    (receiver, fold_id)
                    for receiver in train_receivers
                    for fold_id in fold_ids
                }
                if (
                    outer not in receivers
                    or outer in selection["outer_train_receivers"]
                    or set(selection["outer_train_receivers"])
                    != set(train_receivers)
                    or tuple(selection["outer_train_receivers"]) != train_receivers
                    or set(selection["outer_ground_domain_ids"])
                    != set(all_ground_domains) - {outer_domain}
                    or len(selection["outer_ground_domain_ids"])
                    != len(all_ground_domains) - 1
                    or selection["prepared_cache_scope"] != f"outer_only:{outer}"
                ):
                    return False
                d99_records = []
                stored_d99_candidate_ids = [
                    record["candidate_id"] for record in selection["d99_candidates"]
                ]
                if stored_d99_candidate_ids != expected_d99_candidate_ids:
                    return False
                for record in selection["d99_candidates"]:
                    row_keys = [(row["receiver"], row["fold_id"]) for row in record["rows"]]
                    if (
                        len(row_keys) != len(expected_inner_keys)
                        or len(row_keys) != len(set(row_keys))
                        or set(row_keys) != expected_inner_keys
                        or not all(
                            _verify_base_selection_row(row, outer, outer_domain)
                            and row["pseudo_old"] == fold_lookup[row["fold_id"]]["pseudo_old"]
                            and row["pseudo_new"] == fold_lookup[row["fold_id"]]["pseudo_new"]
                            for row in record["rows"]
                        )
                    ):
                        return False
                    d99_records.append(
                        _record_d99_candidate(
                            record["candidate_id"],
                            record["parameters"],
                            record["rows"],
                            k_shot,
                        )
                    )
                if _jsonable(d99_records) != _jsonable(selection["d99_candidates"]):
                    return False
                winner99 = _winner(d99_records)
                winner99_id = None if winner99 is None else winner99["candidate_id"]
                if winner99_id != selection["selected_d99_candidate_id"]:
                    return False
                d100_records = []
                stored_d100_candidate_ids = [
                    record["candidate_id"] for record in selection["d100_candidates"]
                ]
                expected_d100_candidate_ids = (
                    []
                    if winner99_id is None
                    else expected_d100_candidate_ids_by_d99[winner99_id]
                )
                if stored_d100_candidate_ids != expected_d100_candidate_ids:
                    return False
                for record in selection["d100_candidates"]:
                    row_keys = [(row["receiver"], row["fold_id"]) for row in record["rows"]]
                    if (
                        len(row_keys) != len(expected_inner_keys)
                        or len(row_keys) != len(set(row_keys))
                        or set(row_keys) != expected_inner_keys
                        or not all(
                            _verify_base_selection_row(row, outer, outer_domain)
                            and row["pseudo_old"] == fold_lookup[row["fold_id"]]["pseudo_old"]
                            and row["pseudo_new"] == fold_lookup[row["fold_id"]]["pseudo_new"]
                            for row in record["rows"]
                        )
                    ):
                        return False
                    d100_records.append(
                        _record_d100_candidate(
                            record["candidate_id"], record["parameters"], record["rows"]
                        )
                    )
                if _jsonable(d100_records) != _jsonable(selection["d100_candidates"]):
                    return False
                if winner99 is not None and any(
                    _shared_signature(record["parameters"])
                    != dict(winner99["parameters"])
                    for record in d100_records
                ):
                    return False
                control100 = (
                    None
                    if winner99 is None
                    else _select_d100_control(d100_records, winner99["parameters"])
                )
                winner100_id = (
                    None if control100 is None else control100["effective_control_id"]
                )
                if (
                    _jsonable(control100) != _jsonable(selection["selected_d100_control"])
                    or winner100_id != selection["selected_d100_candidate_id"]
                    or (
                        control100 is not None
                        and control100["source_requested_candidate_id"]
                        != selection["selected_d100_source_requested_candidate_id"]
                    )
                ):
                    return False
                d101_records = []
                stored_d101_candidate_ids = [
                    record["candidate_id"] for record in selection["d101_candidates"]
                ]
                expected_selection_d101_candidate_ids = (
                    [] if winner99_id is None else expected_d101_candidate_ids
                )
                if stored_d101_candidate_ids != expected_selection_d101_candidate_ids:
                    return False
                for record in selection["d101_candidates"]:
                    row_keys = [(row["receiver"], row["fold_id"]) for row in record["rows"]]
                    if (
                        len(row_keys) != len(expected_inner_keys)
                        or len(row_keys) != len(set(row_keys))
                        or set(row_keys) != expected_inner_keys
                        or not all(
                            _verify_joint_row(row, outer, outer_domain, gate_lock)
                            and row["split_name"] == "inner_calibration"
                            and row["receiver"] != outer
                            and row["base_candidate_sha256"] == winner100_id
                            and row["d101_candidate_sha256"] == record["candidate_id"]
                            and row["d100_control_mode"] == control100["control_mode"]
                            and row["pseudo_old"] == fold_lookup[row["fold_id"]]["pseudo_old"]
                            and row["pseudo_new"] == fold_lookup[row["fold_id"]]["pseudo_new"]
                            for row in record["rows"]
                        )
                    ):
                        return False
                    d101_records.append(
                        _record_d101_candidate(
                            record["candidate_id"],
                            record["parameters"],
                            record["rows"],
                            gate_lock,
                            k_shot,
                        )
                    )
                if _jsonable(d101_records) != _jsonable(selection["d101_candidates"]):
                    return False
                winner101 = _winner(d101_records)
                winner101_id = None if winner101 is None else winner101["candidate_id"]
                if winner101_id != selection["selected_d101_candidate_id"]:
                    return False
                expected_status = (
                    "REJECTED_D99"
                    if winner99 is None
                    else "REJECTED_D101_INNER_GATE"
                    if winner101 is None
                    else "NESTED_WINNERS_FROZEN"
                )
                if selection["selection_status"] != expected_status:
                    return False
                if expected_status != "NESTED_WINNERS_FROZEN":
                    all_selected = False
                    continue
                matching = [
                    row
                    for row in result["outer_held_evaluation_rows"]
                    if row["outer_held_receiver"] == outer
                ]
                if len(matching) != len(folds):
                    return False
                for row in matching:
                    if (
                        not _verify_joint_row(row, outer, outer_domain, gate_lock)
                        or row["split_name"] != "outer_held_evaluation"
                        or row["receiver"] != outer
                        or row["base_candidate_sha256"] != winner100_id
                        or row["d101_candidate_sha256"] != winner101_id
                        or row["d100_control_mode"] != control100["control_mode"]
                        or row["pseudo_old"] != fold_lookup[row["fold_id"]]["pseudo_old"]
                        or row["pseudo_new"] != fold_lookup[row["fold_id"]]["pseudo_new"]
                    ):
                        return False
                verified_outer_rows.extend(matching)
            if [row["joint_row_sha256"] for row in verified_outer_rows] != [
                row["joint_row_sha256"] for row in stored_outer_rows
            ]:
                return False
            expected_available = bool(
                all_selected and len(verified_outer_rows) == len(receivers) * len(folds)
            )
            if expected_available != result["all_outer_nested_winners_available"]:
                return False
            expected_gate = (
                _aggregate_joint_rows(verified_outer_rows, gate_lock, k_shot=k_shot)
                if expected_available
                else None
            )
            if _jsonable(expected_gate) != _jsonable(result["outer_hard_gate"]):
                return False
            expected_pass = bool(expected_available and expected_gate and expected_gate["passed"])
            if expected_pass != result["passed"]:
                return False
            scientific_passes.append(expected_pass)
        scientific = bool(all(scientific_passes))
        if scientific != receipt["scientific_phase1_hard_gate_passed"]:
            return False
        expected_status = STATUS_ADMITTED if scientific else STATUS_REJECTED
        return bool(receipt["status"] == expected_status)
    except (KeyError, TypeError, ValueError, D101Phase1LODOError, base.D99D100LODOLockError):
        return False


def predict_formal(*_args: Any, **_kwargs: Any) -> None:
    raise D101Phase1LODOError(
        "D101 Phase1 LODO is diagnostic-only and cannot authorize target/N607 prediction"
    )


__all__ = [
    "D101LODOGateLock",
    "D101Phase1LODOError",
    "d101_candidate_grid",
    "held_quantization_margin_audit",
    "run_phase1_d101_nested_lodo",
    "verify_receipt",
    "predict_formal",
    "current_code_sha256",
]
