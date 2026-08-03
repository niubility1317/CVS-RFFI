"""Deterministic, runner-neutral Phase1 assets for the frozen D127 DA paths.

This module is deliberately narrow.  Callers supply already-frozen source
episodes, checkpoint/tap outputs, and loss callbacks.  It never imports a data
builder, checkpoint, target capsule, runner, scorer, or transport layer.

The persistent outputs are ``QuantizedFSRGAsset`` and
``QuantizedRDHAAsset``.  They contain only typed byte payloads and FP16
scales/statistics.  ``decode`` creates a fresh, no-grad float32 runtime view;
it is not a persistent FP32 sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from cvsrffi.stage2_d127_da_candidates import (
    CANDIDATE_A,
    CANDIDATE_B,
    CANDIDATE_C,
    D127DACandidateError,
    FSRGAsset,
    JOINT_PROJ_INPUT_DIM,
    LAYER_NORM_EPS,
    RANK,
    RDHAAsset,
    RELATIVE_RESIDUAL_BUDGET,
    SUMMARY_DIM,
    TAP_A,
    TAP_B,
    TAP_C,
    class_balanced_support_loss,
)


LBFGS_MAX_ITER = 128
LBFGS_LINE_SEARCH = "strong_wolfe"
_EPS64 = float(torch.finfo(torch.float64).eps)
_SVD_GAP_RELATIVE_TOL = math.sqrt(_EPS64)
_FP16_TINY = float(np.finfo(np.float16).tiny)
_FSRG_TAPS = {CANDIDATE_A: TAP_A, CANDIDATE_B: TAP_B}


class D127Phase1AssetError(D127DACandidateError):
    """Raised when a frozen Phase1 asset contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D127Phase1AssetError(message)


def _finite_float32(value: Tensor, *, name: str, ndim: int | tuple[int, ...]) -> Tensor:
    allowed = (ndim,) if isinstance(ndim, int) else ndim
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.ndim not in allowed
        or int(value.shape[0]) < 1
        or not bool(torch.isfinite(value).all().item())
    ):
        expected = "/".join(str(item) for item in allowed)
        raise D127Phase1AssetError(
            f"{name} must be finite float32 with an allowed ndim of {expected}"
        )
    return value


def _integral_labels(value: Tensor, *, rows: int, device: torch.device, name: str) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 1
        or int(value.shape[0]) != rows
        or value.dtype.is_floating_point
        or value.dtype == torch.bool
        or value.device != device
    ):
        raise D127Phase1AssetError(
            f"{name} must be an integral [N] tensor on the feature device"
        )
    return value


def _class_groups(labels: Tensor, *, expected_k: int, name: str) -> tuple[Tensor, ...]:
    unique, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    if int(unique.numel()) < 2:
        raise D127Phase1AssetError(f"{name} must contain at least two classes")
    if not bool(torch.all(counts == expected_k).item()):
        raise D127Phase1AssetError(
            f"{name} must contain exactly K={expected_k} physical rows per class"
        )
    return tuple(
        torch.nonzero(inverse == index, as_tuple=False).reshape(-1)
        for index in range(int(unique.numel()))
    )


def _same_class_set(left: Tensor, right: Tensor) -> bool:
    return bool(torch.equal(torch.unique(left, sorted=True), torch.unique(right, sorted=True)))


def _id_tuple(value: Sequence[str], *, rows: int, name: str) -> tuple[str, ...]:
    try:
        result = tuple(str(item) for item in value)
    except TypeError as exc:
        raise D127Phase1AssetError(f"{name} must be a sequence of physical IDs") from exc
    if (
        len(result) != rows
        or any(not item for item in result)
        or len(set(result)) != len(result)
    ):
        raise D127Phase1AssetError(
            f"{name} must be nonempty, unique, and aligned to its feature rows"
        )
    return result


def _validate_support_query_ids(
    support_ids: Sequence[str], query_ids: Sequence[str], *, support_rows: int, query_rows: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    support = _id_tuple(support_ids, rows=support_rows, name="support_physical_ids")
    query = _id_tuple(query_ids, rows=query_rows, name="query_physical_ids")
    if set(support).intersection(query):
        raise D127Phase1AssetError(
            "support_physical_ids and query_physical_ids must be physically disjoint"
        )
    return support, query


def _tap_rows(value: Tensor, *, name: str) -> Tensor:
    result = _finite_float32(value, name=name, ndim=(2, 3))
    if int(result.shape[1]) < RANK or (result.ndim == 3 and int(result.shape[2]) < 1):
        raise D127Phase1AssetError(f"{name} must expose at least rank-two finite tap channels")
    return result


def _hidden_rows(value: Tensor, *, name: str) -> Tensor:
    result = _finite_float32(value, name=name, ndim=2)
    if int(result.shape[1]) != JOINT_PROJ_INPUT_DIM:
        raise D127Phase1AssetError(
            f"{name} must have exactly {JOINT_PROJ_INPUT_DIM} hidden dimensions"
        )
    return result


@dataclass(frozen=True, slots=True)
class CanonicalRank2Initialization:
    """Ephemeral float32 initialization derived by a canonical float64 SVD."""

    U: Tensor
    V: Tensor
    singular_values: tuple[float, float]

    def __post_init__(self) -> None:
        U = _finite_float32(self.U, name="initial U", ndim=2)
        V = _finite_float32(self.V, name="initial V", ndim=2)
        _require(
            int(U.shape[1]) == RANK
            and int(V.shape[0]) == RANK
            and int(U.shape[0]) == int(V.shape[1]),
            "canonical initialization must be U[d,2] and V[2,d]",
        )
        _require(U.device.type == "cpu" and V.device.type == "cpu", "canonical initialization must be CPU resident")
        _require(
            all(math.isfinite(float(item)) and float(item) > 0.0 for item in self.singular_values),
            "canonical initialization must retain two positive singular values",
        )
        object.__setattr__(self, "U", U.detach().clone().contiguous())
        object.__setattr__(self, "V", V.detach().clone().contiguous())

    @property
    def dimension(self) -> int:
        return int(self.U.shape[0])


def canonical_receiver_mean_svd(
    receiver_class_means: Tensor, *, dimension: int | None = None
) -> CanonicalRank2Initialization:
    """Create the one permitted rank-two initialization from source means.

    ``receiver_class_means`` is the caller's already aggregated tensor
    ``[receiver, class, feature]``.  The function removes each class's mean
    across receivers, averages the resulting receiver differences equally over
    classes, and runs one CPU float64 SVD.  The largest-absolute coordinate of
    each selected right singular vector is made positive, including a stable
    first-index tie break supplied by ``torch.argmax``.
    """

    if (
        not torch.is_tensor(receiver_class_means)
        or not receiver_class_means.dtype.is_floating_point
        or receiver_class_means.ndim != 3
        or int(receiver_class_means.shape[0]) < 1
        or int(receiver_class_means.shape[1]) < 1
        or int(receiver_class_means.shape[2]) < RANK
        or not bool(torch.isfinite(receiver_class_means).all().item())
    ):
        raise D127Phase1AssetError(
            "receiver_class_means must be finite floating [receiver,class,feature]"
        )
    width = int(receiver_class_means.shape[2])
    if dimension is not None and width != dimension:
        raise D127Phase1AssetError("receiver_class_means feature width drift")
    means = receiver_class_means.detach().to(device="cpu", dtype=torch.float64)
    class_centered = means - means.mean(dim=0, keepdim=True)
    receiver_difference = class_centered.mean(dim=1)
    receiver_difference = receiver_difference - receiver_difference.mean(dim=0, keepdim=True)
    _left, singular, right = torch.linalg.svd(receiver_difference, full_matrices=False)
    if int(singular.numel()) < RANK:
        raise D127Phase1AssetError("receiver-mean difference matrix has effective rank below two")
    lead = float(singular[0].item())
    tolerance = _EPS64 * max(receiver_difference.shape) * max(1.0, lead)
    if float(singular[1].item()) <= tolerance:
        raise D127Phase1AssetError("receiver-mean difference matrix has effective rank below two")
    gap_tolerance = (
        _SVD_GAP_RELATIVE_TOL
        * max(receiver_difference.shape)
        * max(1.0, lead)
    )
    selected_gaps = [float(singular[0].item() - singular[1].item())]
    if int(singular.numel()) > RANK:
        selected_gaps.append(float(singular[1].item() - singular[2].item()))
    if any(gap <= gap_tolerance for gap in selected_gaps):
        raise D127Phase1AssetError(
            "receiver-mean SVD has non-unique top-two singular directions"
        )
    directions = right[:RANK].clone()
    for index in range(RANK):
        row = directions[index]
        pivot = int(torch.argmax(torch.abs(row)).item())
        if float(row[pivot].item()) < 0.0:
            directions[index].mul_(-1.0)
    return CanonicalRank2Initialization(
        U=directions.transpose(0, 1).to(dtype=torch.float32).contiguous(),
        V=directions.to(dtype=torch.float32).contiguous(),
        singular_values=(float(singular[0].item()), float(singular[1].item())),
    )


@dataclass(frozen=True, slots=True)
class FSRGEpisode:
    """One caller-provided, physically separated Phase1 A/B episode."""

    episode_id: str
    receiver_id: str
    k_shot: int
    support_taps: Tensor
    support_labels: Tensor
    query_taps: Tensor
    query_labels: Tensor
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(bool(self.episode_id), "episode_id must be nonempty")
        _require(bool(self.receiver_id), "receiver_id must be nonempty")
        _require(self.k_shot in (1, 5), "only pre-frozen K1 and K5 Phase1 episodes are allowed")
        support = _tap_rows(self.support_taps, name="support_taps")
        query = _tap_rows(self.query_taps, name="query_taps")
        _require(
            int(support.shape[1]) == int(query.shape[1]) and support.ndim == query.ndim,
            "support/query tap layout must match within one episode",
        )
        if support.ndim == 3:
            _require(
                int(support.shape[2]) == int(query.shape[2]),
                "support/query time positions must match within one episode",
            )
        labels = _integral_labels(
            self.support_labels,
            rows=int(support.shape[0]),
            device=support.device,
            name="support_labels",
        )
        query_labels = _integral_labels(
            self.query_labels,
            rows=int(query.shape[0]),
            device=query.device,
            name="query_labels",
        )
        _require(query.device == support.device, "support/query taps must share a device")
        _class_groups(labels, expected_k=self.k_shot, name="support_labels")
        _require(
            _same_class_set(labels, query_labels),
            "query labels must cover exactly the registered support classes in Phase1",
        )
        support_ids, query_ids = _validate_support_query_ids(
            self.support_physical_ids,
            self.query_physical_ids,
            support_rows=int(support.shape[0]),
            query_rows=int(query.shape[0]),
        )
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)

    @property
    def dimension(self) -> int:
        return int(self.support_taps.shape[1])


@dataclass(frozen=True, slots=True)
class RDHAEpisode:
    """One caller-provided, physically separated Phase1 C episode."""

    episode_id: str
    receiver_id: str
    k_shot: int
    support_hidden: Tensor
    support_labels: Tensor
    query_hidden: Tensor
    query_labels: Tensor
    support_physical_ids: tuple[str, ...]
    query_physical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(bool(self.episode_id), "episode_id must be nonempty")
        _require(bool(self.receiver_id), "receiver_id must be nonempty")
        _require(self.k_shot in (1, 5), "only pre-frozen K1 and K5 Phase1 episodes are allowed")
        support = _hidden_rows(self.support_hidden, name="support_hidden")
        query = _hidden_rows(self.query_hidden, name="query_hidden")
        _require(query.device == support.device, "support/query hidden rows must share a device")
        labels = _integral_labels(
            self.support_labels,
            rows=int(support.shape[0]),
            device=support.device,
            name="support_labels",
        )
        query_labels = _integral_labels(
            self.query_labels,
            rows=int(query.shape[0]),
            device=query.device,
            name="query_labels",
        )
        _class_groups(labels, expected_k=self.k_shot, name="support_labels")
        _require(
            _same_class_set(labels, query_labels),
            "query labels must cover exactly the registered support classes in Phase1",
        )
        support_ids, query_ids = _validate_support_query_ids(
            self.support_physical_ids,
            self.query_physical_ids,
            support_rows=int(support.shape[0]),
            query_rows=int(query.shape[0]),
        )
        object.__setattr__(self, "support_physical_ids", support_ids)
        object.__setattr__(self, "query_physical_ids", query_ids)


@dataclass(frozen=True, slots=True)
class FrozenFSRGEpisodes:
    """A lexicographically frozen K1/K5 Phase1 episode contract for A or B."""

    episodes: tuple[FSRGEpisode, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.episodes, key=lambda item: item.episode_id))
        _require(bool(ordered), "at least one Phase1 episode is required")
        _require(
            len({item.episode_id for item in ordered}) == len(ordered),
            "Phase1 episode IDs must be unique",
        )
        dimension = ordered[0].dimension
        device = ordered[0].support_taps.device
        for item in ordered:
            _require(item.dimension == dimension, "all FSRG episodes must share one tap dimension")
            _require(item.support_taps.device == device, "all FSRG episodes must share one device")
        _validate_episode_coverage(ordered)
        object.__setattr__(self, "episodes", ordered)

    @property
    def dimension(self) -> int:
        return self.episodes[0].dimension


@dataclass(frozen=True, slots=True)
class FrozenRDHAEpisodes:
    """A lexicographically frozen K1/K5 Phase1 episode contract for C."""

    episodes: tuple[RDHAEpisode, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.episodes, key=lambda item: item.episode_id))
        _require(bool(ordered), "at least one Phase1 episode is required")
        _require(
            len({item.episode_id for item in ordered}) == len(ordered),
            "Phase1 episode IDs must be unique",
        )
        device = ordered[0].support_hidden.device
        for item in ordered:
            _require(item.support_hidden.device == device, "all RDHA episodes must share one device")
        _validate_episode_coverage(ordered)
        object.__setattr__(self, "episodes", ordered)


def _validate_episode_coverage(episodes: Sequence[FSRGEpisode | RDHAEpisode]) -> None:
    by_receiver: dict[str, set[int]] = {}
    all_support_ids: set[str] = set()
    all_query_ids: set[str] = set()
    for item in episodes:
        by_receiver.setdefault(item.receiver_id, set()).add(item.k_shot)
        all_support_ids.update(item.support_physical_ids)
        all_query_ids.update(item.query_physical_ids)
    _require(
        all(values == {1, 5} for values in by_receiver.values()),
        "each inner receiver must contribute both frozen K1 and K5 episodes",
    )
    _require(
        not all_support_ids.intersection(all_query_ids),
        "frozen Phase1 support/query physical-ID sets must be globally disjoint",
    )


@dataclass(frozen=True, slots=True)
class FSRGLossCallbacks:
    """Loss-only hooks supplied by the Phase1 checkpoint/tap caller.

    Both callbacks receive the immutable full episode as their first argument
    and return per-sample losses.  Keeping the episode identity at the
    callback boundary is deliberate: a real-checkpoint bridge can bind its
    source raw-IQ support/query rows to exactly that episode rather than
    accepting an unscoped tensor callback.  This asset layer still applies the
    prescribed equal-class and equal-receiver/K reductions; it does not know
    qKNN, a checkpoint, or a source dataset.
    """

    support_per_sample: Callable[[FSRGEpisode, Tensor], Tensor]
    outer_query_per_sample: Callable[[FSRGEpisode, Tensor, Tensor], Tensor]

    def __post_init__(self) -> None:
        _require(callable(self.support_per_sample), "support_per_sample must be callable")
        _require(callable(self.outer_query_per_sample), "outer_query_per_sample must be callable")


RDHALossCallback = Callable[[RDHAEpisode, Tensor, Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class FSRGSealedStatistics:
    """Build-time float32 statistics; quantize before persisting the asset."""

    rho: float
    d_f_diag: Tensor

    def __post_init__(self) -> None:
        _require(math.isfinite(float(self.rho)) and float(self.rho) > 0.0, "rho must be finite and positive")
        diagonal = _finite_float32(self.d_f_diag, name="d_f_diag", ndim=1)
        _require(int(diagonal.numel()) == RANK and bool(torch.all(diagonal > 0.0).item()), "d_f_diag must be positive rank two")
        object.__setattr__(self, "rho", float(self.rho))
        object.__setattr__(self, "d_f_diag", diagonal.detach().clone().contiguous())


@dataclass(frozen=True, slots=True)
class RDHASealedStatistics:
    """Build-time C standardization/budget statistics; quantize before deploy."""

    mean_p1: Tensor
    std_p1: Tensor
    a_max: float

    def __post_init__(self) -> None:
        mean = _finite_float32(self.mean_p1, name="mean_p1", ndim=1)
        std = _finite_float32(self.std_p1, name="std_p1", ndim=1)
        _require(int(mean.numel()) == SUMMARY_DIM and int(std.numel()) == SUMMARY_DIM, "RDHA statistics must have width five")
        _require(bool(torch.all(std > 0.0).item()), "RDHA std_p1 must be positive")
        _require(math.isfinite(float(self.a_max)) and float(self.a_max) > 0.0, "RDHA a_max must be finite and positive")
        object.__setattr__(self, "mean_p1", mean.detach().clone().contiguous())
        object.__setattr__(self, "std_p1", std.detach().clone().contiguous())
        object.__setattr__(self, "a_max", float(self.a_max))


def _fsrg_residual(taps: Tensor, U: Tensor, V: Tensor, a: Tensor) -> Tensor:
    if taps.ndim == 2:
        response = torch.tanh(taps @ V.transpose(0, 1))
        return taps + ((response * a) @ U.transpose(0, 1))
    response = torch.tanh(torch.einsum("rd,ndt->nrt", V, taps))
    residual = torch.einsum("dr,nrt->ndt", U, response * a.reshape(1, RANK, 1))
    return taps + residual


def _layer_normalize(hidden: Tensor) -> Tensor:
    return F.layer_norm(
        hidden,
        normalized_shape=(JOINT_PROJ_INPUT_DIM,),
        weight=None,
        bias=None,
        eps=LAYER_NORM_EPS,
    )


def _rdah_summary(hidden: Tensor, labels: Tensor, V: Tensor) -> Tensor:
    groups = _class_groups(labels, expected_k=_infer_equal_k(labels), name="support_labels")
    response = torch.tanh(_layer_normalize(hidden) @ V.transpose(0, 1))
    means = torch.stack([response.index_select(0, group).mean(dim=0) for group in groups])
    mean = means.mean(dim=0)
    centered = means - mean
    covariance = centered.transpose(0, 1) @ centered / float(len(groups))
    result = torch.stack(
        (mean[0], mean[1], covariance[0, 0], covariance[0, 1], covariance[1, 1])
    )
    if not bool(torch.isfinite(result).all().item()):
        raise D127Phase1AssetError("RDHA summary is non-finite")
    return result


def _infer_equal_k(labels: Tensor) -> int:
    _unique, counts = torch.unique(labels, sorted=True, return_counts=True)
    if int(counts.numel()) < 2 or not bool(torch.all(counts == counts[0]).item()):
        raise D127Phase1AssetError("support labels must contain at least two equal-K classes")
    return int(counts[0].item())


def _validate_loss(value: Tensor, *, rows: int, name: str, device: torch.device) -> Tensor:
    if (
        not torch.is_tensor(value)
        or value.dtype != torch.float32
        or value.ndim != 1
        or int(value.numel()) != rows
        or value.device != device
        or not bool(torch.isfinite(value).all().item())
    ):
        raise D127Phase1AssetError(
            f"{name} must return finite float32 per-sample losses aligned to its rows"
        )
    return value


def _episode_weights(episodes: Sequence[FSRGEpisode | RDHAEpisode]) -> tuple[float, ...]:
    receivers = tuple(sorted({item.receiver_id for item in episodes}))
    _require(bool(receivers), "at least one receiver is required")
    grouped: dict[tuple[str, int], int] = {}
    for item in episodes:
        grouped[(item.receiver_id, item.k_shot)] = grouped.get(
            (item.receiver_id, item.k_shot), 0
        ) + 1
    weight: list[float] = []
    for item in episodes:
        count = grouped[(item.receiver_id, item.k_shot)]
        weight.append(1.0 / float(len(receivers) * 2 * count))
    return tuple(weight)


def _balanced_episode_mean(
    episodes: Sequence[FSRGEpisode | RDHAEpisode], values: Sequence[Tensor]
) -> Tensor:
    _require(len(episodes) == len(values) and bool(values), "episode/value sequence drift")
    weights = _episode_weights(episodes)
    result = values[0] * weights[0]
    for value, weight in zip(values[1:], weights[1:]):
        result = result + value * weight
    return result


def _balanced_physical_norm_median(
    episodes: Sequence[FSRGEpisode | RDHAEpisode], *, hidden: bool
) -> float:
    """A deterministic weighted median with equal receiver/K/episode/class mass."""

    weights_by_episode = _episode_weights(episodes)
    values: list[Tensor] = []
    weights: list[Tensor] = []
    for episode, episode_weight in zip(episodes, weights_by_episode):
        rows = episode.support_hidden if hidden else episode.support_taps
        labels = episode.support_labels
        k = episode.k_shot
        groups = _class_groups(labels, expected_k=k, name="support_labels")
        norms = torch.linalg.vector_norm(
            rows.detach().to(device="cpu", dtype=torch.float64).reshape(int(rows.shape[0]), -1),
            dim=1,
        )
        per_sample_weight = episode_weight / float(len(groups) * k)
        for group in groups:
            # Labels and norms may originate on an accelerator; the index is
            # copied only into the source-side build calculation.
            values.append(norms.index_select(0, group.detach().to(device="cpu")))
            weights.append(
                torch.full(
                    (int(group.numel()),),
                    per_sample_weight,
                    dtype=torch.float64,
                    device="cpu",
                )
            )
    all_values = torch.cat(values)
    all_weights = torch.cat(weights)
    ordered_values, order = torch.sort(all_values, stable=True)
    ordered_weights = all_weights.index_select(0, order)
    cumulative = torch.cumsum(ordered_weights, dim=0)
    index = int(torch.nonzero(cumulative >= 0.5, as_tuple=False)[0].item())
    return float(ordered_values[index].item())


def _initialization_on(
    initialization: CanonicalRank2Initialization, *, device: torch.device, dimension: int
) -> tuple[Tensor, Tensor]:
    _require(initialization.dimension == dimension, "canonical initialization dimension drift")
    return (
        initialization.U.to(device=device, dtype=torch.float32).detach().clone(),
        initialization.V.to(device=device, dtype=torch.float32).detach().clone(),
    )


def _fsrg_support_gradient_rows(
    episode: FSRGEpisode,
    U: Tensor,
    V: Tensor,
    support_per_sample: Callable[[FSRGEpisode, Tensor], Tensor],
) -> Tensor:
    a0 = torch.zeros(RANK, dtype=torch.float32, device=episode.support_taps.device, requires_grad=True)
    adapted = _fsrg_residual(episode.support_taps.detach(), U.detach(), V.detach(), a0)
    loss = _validate_loss(
        support_per_sample(episode, adapted),
        rows=int(adapted.shape[0]),
        name="support_per_sample",
        device=adapted.device,
    )
    # Calling grad separately preserves the exact per-physical-row gradient
    # needed by the frozen diagonal variance construction.
    gradients: list[Tensor] = []
    for index in range(int(loss.numel())):
        try:
            gradient = torch.autograd.grad(
                loss[index], a0, retain_graph=index + 1 < int(loss.numel()), create_graph=False
            )[0]
        except RuntimeError as exc:
            raise D127Phase1AssetError(
                "support_per_sample is not differentiably bound to the rank-two state"
            ) from exc
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise D127Phase1AssetError("per-physical support gradient is missing or non-finite")
        gradients.append(gradient.detach().to(dtype=torch.float64))
    result = torch.stack(gradients)
    if not bool(torch.any(torch.abs(result) > 0.0).item()):
        raise D127Phase1AssetError("support-gradient fixture is identically zero")
    return result


def seal_fsrg_statistics(
    episodes: FrozenFSRGEpisodes,
    initialization: CanonicalRank2Initialization,
    support_per_sample: Callable[[FSRGEpisode, Tensor], Tensor],
) -> FSRGSealedStatistics:
    """Seal A/B's ``rho`` and ``D_F`` only from Phase1 inner episodes.

    The rank-two support gradients use fixed initial U/V and are accumulated in
    float64.  Their variance is averaged receiver-equally; the numerical floor
    is precisely ``eps64 * max(1, mean(D_F_raw))``.  ``rho`` is the fixed
    five-percent median norm of the provided inner taps and is never inferred
    from target support.
    """

    _require(callable(support_per_sample), "support_per_sample must be callable")
    first = episodes.episodes[0]
    U, V = _initialization_on(
        initialization, device=first.support_taps.device, dimension=episodes.dimension
    )
    variances: list[Tensor] = []
    for episode in episodes.episodes:
        gradients = _fsrg_support_gradient_rows(episode, U, V, support_per_sample)
        # Equal K within an episode makes this an equal-class sample variance.
        variance = gradients.square().mean(dim=0) - gradients.mean(dim=0).square()
        variances.append(variance)
    weights = _episode_weights(episodes.episodes)
    raw = sum(
        (variance * weight for variance, weight in zip(variances, weights)),
        torch.zeros(RANK, dtype=torch.float64, device=variances[0].device),
    )
    floor = _EPS64 * max(1.0, float(raw.mean().item()))
    d_f = torch.clamp(raw, min=floor)
    _require(bool(torch.isfinite(d_f).all().item()), "D_F became non-finite")
    rho = RELATIVE_RESIDUAL_BUDGET * _balanced_physical_norm_median(
        episodes.episodes, hidden=False
    )
    _require(math.isfinite(rho) and rho > 0.0, "Phase1 inner-tap median produced zero rho")
    return FSRGSealedStatistics(
        rho=rho,
        d_f_diag=d_f.to(dtype=torch.float32, device=first.support_taps.device),
    )


def seal_rdah_statistics(
    episodes: FrozenRDHAEpisodes, initialization: CanonicalRank2Initialization
) -> RDHASealedStatistics:
    """Seal C's Phase1-only summary normalization and residual budget."""

    first = episodes.episodes[0]
    _U, V = _initialization_on(
        initialization, device=first.support_hidden.device, dimension=JOINT_PROJ_INPUT_DIM
    )
    summaries = torch.stack(
        [
            _rdah_summary(episode.support_hidden.detach(), episode.support_labels, V.detach()).to(
                dtype=torch.float64
            )
            for episode in episodes.episodes
        ]
    )
    weights = torch.tensor(
        _episode_weights(episodes.episodes), dtype=torch.float64, device=summaries.device
    ).reshape(-1, 1)
    mean = (summaries * weights).sum(dim=0)
    variance = ((summaries - mean).square() * weights).sum(dim=0)
    raw_std = torch.sqrt(torch.clamp(variance, min=0.0))
    floor = _EPS64 * torch.maximum(torch.ones_like(mean), torch.abs(mean))
    std = torch.maximum(raw_std, floor)
    a_max = (
        RELATIVE_RESIDUAL_BUDGET
        * _balanced_physical_norm_median(episodes.episodes, hidden=True)
        / math.sqrt(2.0)
    )
    _require(math.isfinite(a_max) and a_max > 0.0, "Phase1 inner-hidden median produced zero a_max")
    return RDHASealedStatistics(
        mean_p1=mean.to(dtype=torch.float32),
        std_p1=std.to(dtype=torch.float32),
        a_max=a_max,
    )


@dataclass(frozen=True, slots=True)
class DeterministicLBFGSReceipt:
    """Non-performance receipt for the single frozen full-batch optimization."""

    parameter_names: tuple[str, ...]
    max_iter: int
    line_search_fn: str
    initialization_count: int
    closure_calls: int
    internal_iterations: int
    initial_loss: float
    final_loss: float
    initial_gradient_norm: float
    initial_parameter_gradient_norms: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(
            self.max_iter == LBFGS_MAX_ITER
            and self.line_search_fn == LBFGS_LINE_SEARCH
            and self.initialization_count == 1,
            "L-BFGS budget/line-search/initialization contract drift",
        )
        _require(bool(self.parameter_names), "optimizer must expose trainable parameters")
        _require(
            self.closure_calls >= 1 and self.internal_iterations >= 1,
            "L-BFGS did not execute a full-batch closure",
        )
        numeric = (
            self.initial_loss,
            self.final_loss,
            self.initial_gradient_norm,
            *self.initial_parameter_gradient_norms,
        )
        _require(all(math.isfinite(float(value)) for value in numeric), "L-BFGS receipt contains non-finite values")
        _require(self.initial_gradient_norm > 0.0, "Phase1 outer optimization has zero initial gradient")


@dataclass(frozen=True, slots=True)
class FSRGPhase1TrainingResult:
    """Ephemeral build result.  Persist only ``quantize_fsrg_asset(asset)``."""

    asset: FSRGAsset
    statistics: FSRGSealedStatistics
    receipt: DeterministicLBFGSReceipt


@dataclass(frozen=True, slots=True)
class RDHAPhase1TrainingResult:
    """Ephemeral build result.  Persist only ``quantize_rdah_asset(asset)``."""

    asset: RDHAAsset
    statistics: RDHASealedStatistics
    receipt: DeterministicLBFGSReceipt


def _fsrg_support_state(
    episode: FSRGEpisode,
    U: Tensor,
    V: Tensor,
    statistics: FSRGSealedStatistics,
    support_per_sample: Callable[[FSRGEpisode, Tensor], Tensor],
) -> Tensor:
    """First-order ``S -> a1`` state; deliberately no U/V second-order graph."""

    a0 = torch.zeros(RANK, dtype=torch.float32, device=episode.support_taps.device, requires_grad=True)
    support = _fsrg_residual(episode.support_taps.detach(), U.detach(), V.detach(), a0)
    losses = _validate_loss(
        support_per_sample(episode, support),
        rows=int(support.shape[0]),
        name="support_per_sample",
        device=support.device,
    )
    try:
        support_loss = class_balanced_support_loss(losses, episode.support_labels)
        gradient = torch.autograd.grad(
            support_loss, a0, create_graph=False, retain_graph=False, allow_unused=False
        )[0]
    except RuntimeError as exc:
        raise D127Phase1AssetError(
            "support_per_sample is not differentiably bound to the rank-two state"
        ) from exc
    if (
        gradient is None
        or not bool(torch.isfinite(gradient).all().item())
        or not bool(torch.any(torch.abs(gradient) > 0.0).item())
    ):
        raise D127Phase1AssetError("Phase1 FSRG support state is zero or non-finite")
    # The model forward is float32, while the frozen projection arithmetic is
    # accumulated in float64 exactly as specified by the Phase1 contract.
    raw64 = -gradient.detach().to(dtype=torch.float64) / torch.sqrt(
        statistics.d_f_diag.detach().to(dtype=torch.float64)
    )
    a_max = statistics.rho / math.sqrt(2.0)
    projected64 = torch.clamp(raw64, min=-a_max, max=a_max)
    norm64 = torch.linalg.vector_norm(projected64)
    if float(norm64.item()) > statistics.rho:
        projected64 = projected64 * (statistics.rho / norm64)
    a = projected64.to(dtype=torch.float32).detach()
    if not bool(torch.any(torch.abs(a) > 0.0).item()):
        raise D127Phase1AssetError("Phase1 FSRG projected support state is zero")
    return a


def _fsrg_outer_episode_loss(
    episode: FSRGEpisode,
    U: Tensor,
    V: Tensor,
    statistics: FSRGSealedStatistics,
    callbacks: FSRGLossCallbacks,
) -> Tensor:
    a = _fsrg_support_state(episode, U, V, statistics, callbacks.support_per_sample)
    adapted_support = _fsrg_residual(episode.support_taps, U, V, a)
    adapted_query = _fsrg_residual(episode.query_taps, U, V, a)
    losses = _validate_loss(
        callbacks.outer_query_per_sample(episode, adapted_support, adapted_query),
        rows=int(adapted_query.shape[0]),
        name="outer_query_per_sample",
        device=adapted_query.device,
    )
    return class_balanced_support_loss(losses, episode.query_labels)


def _rdah_adapt(hidden: Tensor, U: Tensor, V: Tensor, a: Tensor) -> Tensor:
    response = torch.tanh(_layer_normalize(hidden) @ V.transpose(0, 1))
    return hidden + ((response * a) @ U.transpose(0, 1))


def _rdah_outer_episode_loss(
    episode: RDHAEpisode,
    U: Tensor,
    V: Tensor,
    Q: Tensor,
    b: Tensor,
    statistics: RDHASealedStatistics,
    outer_query_per_sample: RDHALossCallback,
) -> Tensor:
    summary = _rdah_summary(episode.support_hidden, episode.support_labels, V)
    standardized = (summary - statistics.mean_p1.detach()) / statistics.std_p1.detach()
    a = statistics.a_max * torch.tanh(Q @ standardized + b)
    adapted_support = _rdah_adapt(episode.support_hidden, U, V, a)
    adapted_query = _rdah_adapt(episode.query_hidden, U, V, a)
    losses = _validate_loss(
        outer_query_per_sample(episode, adapted_support, adapted_query),
        rows=int(adapted_query.shape[0]),
        name="outer_query_per_sample",
        device=adapted_query.device,
    )
    return class_balanced_support_loss(losses, episode.query_labels)


def _optimize_full_batch(
    *,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    objective: Callable[[], Tensor],
) -> DeterministicLBFGSReceipt:
    """Run exactly one deterministic full-batch L-BFGS initialization."""

    parameters = [item[1] for item in named_parameters]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=LBFGS_MAX_ITER,
        max_eval=None,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=100,
        line_search_fn=LBFGS_LINE_SEARCH,
    )
    closure_calls = 0
    initial_loss: float | None = None
    initial_gradient_norm: float | None = None
    initial_parameter_norms: tuple[float, ...] | None = None

    def closure() -> Tensor:
        nonlocal closure_calls, initial_loss, initial_gradient_norm, initial_parameter_norms
        optimizer.zero_grad(set_to_none=True)
        value = objective()
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or value.ndim != 0
            or not bool(torch.isfinite(value).item())
        ):
            raise D127Phase1AssetError("full-batch outer objective must be a finite float32 scalar")
        value.backward()
        parameter_norms = tuple(
            0.0
            if parameter.grad is None
            else float(torch.linalg.vector_norm(parameter.grad.detach()).item())
            for parameter in parameters
        )
        norm = math.sqrt(sum(item * item for item in parameter_norms))
        if closure_calls == 0:
            initial_loss = float(value.detach().item())
            initial_gradient_norm = norm
            initial_parameter_norms = parameter_norms
        closure_calls += 1
        return value

    optimizer.step(closure)
    _require(initial_loss is not None and initial_gradient_norm is not None and initial_parameter_norms is not None, "L-BFGS did not evaluate its initial full batch")
    with torch.enable_grad():
        final_loss = objective()
    if not bool(torch.isfinite(final_loss).item()):
        raise D127Phase1AssetError("L-BFGS final full-batch loss is non-finite")
    internal_iterations = int(optimizer.state[parameters[0]].get("n_iter", 0))
    return DeterministicLBFGSReceipt(
        parameter_names=tuple(item[0] for item in named_parameters),
        max_iter=LBFGS_MAX_ITER,
        line_search_fn=LBFGS_LINE_SEARCH,
        initialization_count=1,
        closure_calls=closure_calls,
        internal_iterations=internal_iterations,
        initial_loss=initial_loss,
        final_loss=float(final_loss.detach().item()),
        initial_gradient_norm=initial_gradient_norm,
        initial_parameter_gradient_norms=initial_parameter_norms,
    )


def train_fsrg_phase1_asset(
    *,
    candidate_id: str,
    episodes: FrozenFSRGEpisodes,
    initialization: CanonicalRank2Initialization,
    callbacks: FSRGLossCallbacks,
) -> FSRGPhase1TrainingResult:
    """Train one frozen A/B asset with a single 128-iteration L-BFGS budget."""

    tap_name = _FSRG_TAPS.get(candidate_id)
    if tap_name is None:
        raise D127Phase1AssetError("FSRG training requires frozen candidate A or B")
    first = episodes.episodes[0]
    U0, V0 = _initialization_on(
        initialization, device=first.support_taps.device, dimension=episodes.dimension
    )
    statistics = seal_fsrg_statistics(episodes, initialization, callbacks.support_per_sample)
    U = torch.nn.Parameter(U0)
    V = torch.nn.Parameter(V0)

    def objective() -> Tensor:
        losses = [
            _fsrg_outer_episode_loss(episode, U, V, statistics, callbacks)
            for episode in episodes.episodes
        ]
        return _balanced_episode_mean(episodes.episodes, losses)

    receipt = _optimize_full_batch(
        named_parameters=(("U", U), ("V", V)), objective=objective
    )
    asset = FSRGAsset(
        candidate_id=candidate_id,
        tap_name=tap_name,
        U=U.detach(),
        V=V.detach(),
        d_f_diag=statistics.d_f_diag.detach(),
        rho=statistics.rho,
    )
    return FSRGPhase1TrainingResult(asset=asset, statistics=statistics, receipt=receipt)


def train_rdah_phase1_asset(
    *,
    episodes: FrozenRDHAEpisodes,
    initialization: CanonicalRank2Initialization,
    outer_query_per_sample: RDHALossCallback,
) -> RDHAPhase1TrainingResult:
    """Train candidate C's U/V/Q/b with one frozen 128-iteration L-BFGS call."""

    _require(callable(outer_query_per_sample), "outer_query_per_sample must be callable")
    first = episodes.episodes[0]
    U0, V0 = _initialization_on(
        initialization, device=first.support_hidden.device, dimension=JOINT_PROJ_INPUT_DIM
    )
    statistics = seal_rdah_statistics(episodes, initialization)
    Q0 = torch.zeros((RANK, SUMMARY_DIM), dtype=torch.float32, device=first.support_hidden.device)
    Q0[0, 0] = 1.0
    Q0[1, 1] = 1.0
    U = torch.nn.Parameter(U0)
    V = torch.nn.Parameter(V0)
    Q = torch.nn.Parameter(Q0)
    b = torch.nn.Parameter(torch.zeros(RANK, dtype=torch.float32, device=first.support_hidden.device))

    def objective() -> Tensor:
        losses = [
            _rdah_outer_episode_loss(
                episode, U, V, Q, b, statistics, outer_query_per_sample
            )
            for episode in episodes.episodes
        ]
        return _balanced_episode_mean(episodes.episodes, losses)

    receipt = _optimize_full_batch(
        named_parameters=(("U", U), ("V", V), ("Q", Q), ("b", b)),
        objective=objective,
    )
    asset = RDHAAsset(
        U=U.detach(),
        V=V.detach(),
        Q=Q.detach(),
        b=b.detach(),
        mean_p1=statistics.mean_p1.detach(),
        std_p1=statistics.std_p1.detach(),
        a_max=statistics.a_max,
        candidate_id=CANDIDATE_C,
        tap_name=TAP_C,
    )
    return RDHAPhase1TrainingResult(asset=asset, statistics=statistics, receipt=receipt)


@dataclass(frozen=True, slots=True)
class FP16Buffer:
    """A fixed-width, little-endian FP16 byte field with no tensor sidecar."""

    width: int
    data: bytes

    def __post_init__(self) -> None:
        _require(type(self.width) is int and self.width >= 1, "FP16 width must be positive")
        _require(isinstance(self.data, bytes) and len(self.data) == 2 * self.width, "FP16 byte length drift")
        values = np.frombuffer(self.data, dtype="<f2")
        _require(np.isfinite(values).all(), "FP16 buffer must be finite")

    @classmethod
    def from_tensor(
        cls, value: Tensor, *, name: str, require_positive: bool = False
    ) -> "FP16Buffer":
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
            or value.ndim != 1
            or int(value.numel()) < 1
            or not bool(torch.isfinite(value).all().item())
        ):
            raise D127Phase1AssetError(f"{name} must be a finite floating vector")
        source = value.detach().to(device="cpu", dtype=torch.float32).numpy()
        if require_positive:
            if not bool(np.all(source > 0.0)):
                raise D127Phase1AssetError(f"{name} must be strictly positive")
        encoded = source.astype("<f2")
        if not np.isfinite(encoded).all():
            raise D127Phase1AssetError(f"{name} overflows the frozen FP16 layout")
        if require_positive and not bool(np.all(encoded > 0.0)):
            raise D127Phase1AssetError(
                f"{name} is not representable as positive finite FP16"
            )
        return cls(width=int(encoded.size), data=encoded.tobytes(order="C"))

    def decode(self, *, device: torch.device | str = "cpu") -> Tensor:
        values = np.frombuffer(self.data, dtype="<f2").astype(np.float32, copy=True)
        return torch.from_numpy(values).to(device=device, dtype=torch.float32).detach()

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def require_positive(self, *, name: str) -> None:
        values = np.frombuffer(self.data, dtype="<f2")
        if not bool(np.all(values > 0.0)):
            raise D127Phase1AssetError(f"{name} must be strictly positive in FP16")


@dataclass(frozen=True, slots=True)
class SymmetricInt8Matrix:
    """Symmetric INT8 matrix with the frozen rank-axis scale layout."""

    shape: tuple[int, int]
    group_axis: Literal["column", "row"]
    codes: bytes
    scales: FP16Buffer

    def __post_init__(self) -> None:
        rows, columns = self.shape
        _require(
            type(rows) is int and type(columns) is int and rows >= 1 and columns >= 1,
            "INT8 matrix shape must be positive integers",
        )
        _require(self.group_axis in {"column", "row"}, "INT8 matrix group axis drift")
        _require(isinstance(self.codes, bytes) and len(self.codes) == rows * columns, "INT8 matrix code length drift")
        expected_groups = columns if self.group_axis == "column" else rows
        _require(self.scales.width == expected_groups, "INT8 matrix scale-group count drift")
        self.scales.require_positive(name="INT8 matrix scales")
        codes = np.frombuffer(self.codes, dtype=np.int8)
        _require(not bool(np.any(codes == -128)), "symmetric INT8 may not emit -128")

    @classmethod
    def from_tensor(
        cls, value: Tensor, *, group_axis: Literal["column", "row"], name: str
    ) -> "SymmetricInt8Matrix":
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
            or value.ndim != 2
            or int(value.shape[0]) < 1
            or int(value.shape[1]) < 1
            or not bool(torch.isfinite(value).all().item())
        ):
            raise D127Phase1AssetError(f"{name} must be a finite floating matrix")
        data = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        maxima = np.max(np.abs(data), axis=0 if group_axis == "column" else 1)
        scale64 = np.maximum(maxima / 127.0, _FP16_TINY)
        scale16 = np.asarray(scale64, dtype="<f2")
        if not np.isfinite(scale16).all() or not np.all(scale16 > 0.0):
            raise D127Phase1AssetError(f"{name} cannot be represented by finite FP16 scales")
        scale32 = scale16.astype(np.float32)
        divisor = scale32.reshape(1, -1) if group_axis == "column" else scale32.reshape(-1, 1)
        signed = np.sign(data) * np.floor(np.abs(data / divisor) + 0.5)
        codes = np.clip(signed, -127.0, 127.0).astype(np.int8)
        return cls(
            shape=(int(data.shape[0]), int(data.shape[1])),
            group_axis=group_axis,
            codes=codes.tobytes(order="C"),
            scales=FP16Buffer(width=int(scale16.size), data=scale16.tobytes(order="C")),
        )

    def decode(self, *, device: torch.device | str = "cpu") -> Tensor:
        rows, columns = self.shape
        codes = np.frombuffer(self.codes, dtype=np.int8).reshape(rows, columns).astype(np.float32)
        scales = np.frombuffer(self.scales.data, dtype="<f2").astype(np.float32)
        scale = scales.reshape(1, -1) if self.group_axis == "column" else scales.reshape(-1, 1)
        return torch.from_numpy((codes * scale).astype(np.float32, copy=False)).to(
            device=device, dtype=torch.float32
        ).detach()

    @property
    def nbytes(self) -> int:
        return len(self.codes) + self.scales.nbytes


@dataclass(frozen=True, slots=True)
class SymmetricInt8Vector:
    """A symmetric INT8 vector with one whole-vector FP16 scale."""

    width: int
    codes: bytes
    scale: FP16Buffer

    def __post_init__(self) -> None:
        _require(type(self.width) is int and self.width >= 1, "INT8 vector width must be positive")
        _require(isinstance(self.codes, bytes) and len(self.codes) == self.width, "INT8 vector code length drift")
        _require(self.scale.width == 1, "INT8 vector must retain exactly one FP16 scale")
        self.scale.require_positive(name="INT8 vector scale")
        _require(
            not bool(np.any(np.frombuffer(self.codes, dtype=np.int8) == -128)),
            "symmetric INT8 may not emit -128",
        )

    @classmethod
    def from_tensor(cls, value: Tensor, *, name: str) -> "SymmetricInt8Vector":
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
            or value.ndim != 1
            or int(value.numel()) < 1
            or not bool(torch.isfinite(value).all().item())
        ):
            raise D127Phase1AssetError(f"{name} must be a finite floating vector")
        data = value.detach().to(device="cpu", dtype=torch.float64).numpy()
        scale64 = max(float(np.max(np.abs(data))) / 127.0, _FP16_TINY)
        scale16 = np.asarray([scale64], dtype="<f2")
        if not np.isfinite(scale16).all() or not bool(scale16[0] > 0.0):
            raise D127Phase1AssetError(f"{name} cannot be represented by a finite FP16 scale")
        divisor = float(scale16.astype(np.float32)[0])
        signed = np.sign(data) * np.floor(np.abs(data / divisor) + 0.5)
        codes = np.clip(signed, -127.0, 127.0).astype(np.int8)
        return cls(
            width=int(data.size),
            codes=codes.tobytes(order="C"),
            scale=FP16Buffer(width=1, data=scale16.tobytes(order="C")),
        )

    def decode(self, *, device: torch.device | str = "cpu") -> Tensor:
        codes = np.frombuffer(self.codes, dtype=np.int8).astype(np.float32)
        scale = float(np.frombuffer(self.scale.data, dtype="<f2").astype(np.float32)[0])
        return torch.from_numpy((codes * scale).astype(np.float32, copy=False)).to(
            device=device, dtype=torch.float32
        ).detach()

    @property
    def nbytes(self) -> int:
        return len(self.codes) + self.scale.nbytes


@dataclass(frozen=True, slots=True)
class QuantizedFSRGAsset:
    """Persistent A/B state: four INT8d terms plus exactly fourteen bytes."""

    candidate_id: str
    tap_name: str
    U: SymmetricInt8Matrix
    V: SymmetricInt8Matrix
    d_f_diag: FP16Buffer
    rho: FP16Buffer

    def __post_init__(self) -> None:
        _require(_FSRG_TAPS.get(self.candidate_id) == self.tap_name, "quantized FSRG candidate/tap binding drift")
        _require(self.U.group_axis == "column" and self.V.group_axis == "row", "FSRG rank-axis quantization layout drift")
        _require(
            self.U.shape[1] == RANK
            and self.V.shape[0] == RANK
            and self.U.shape[0] == self.V.shape[1],
            "quantized FSRG U/V shape drift",
        )
        _require(self.d_f_diag.width == RANK and self.rho.width == 1, "quantized FSRG statistic width drift")
        self.d_f_diag.require_positive(name="quantized D_F")
        self.rho.require_positive(name="quantized rho")
        _require(self.numeric_payload_bytes == 4 * self.dimension + 14, "quantized FSRG byte formula drift")

    @property
    def dimension(self) -> int:
        return self.U.shape[0]

    @property
    def numeric_payload_bytes(self) -> int:
        return self.U.nbytes + self.V.nbytes + self.d_f_diag.nbytes + self.rho.nbytes

    @property
    def persistent_fp32_sidecar(self) -> bool:
        return False

    def decode(self, *, device: torch.device | str = "cpu") -> FSRGAsset:
        diagonal = self.d_f_diag.decode(device=device)
        rho = float(self.rho.decode(device=device)[0].item())
        return FSRGAsset(
            candidate_id=self.candidate_id,
            tap_name=self.tap_name,
            U=self.U.decode(device=device),
            V=self.V.decode(device=device),
            d_f_diag=diagonal,
            rho=rho,
        )


@dataclass(frozen=True, slots=True)
class QuantizedRDHAAsset:
    """Persistent C state with the exact 1328-byte numeric payload."""

    U: SymmetricInt8Matrix
    V: SymmetricInt8Matrix
    Q: SymmetricInt8Matrix
    b: SymmetricInt8Vector
    mean_p1: FP16Buffer
    std_p1: FP16Buffer
    a_max: FP16Buffer
    candidate_id: str = CANDIDATE_C
    tap_name: str = TAP_C

    def __post_init__(self) -> None:
        _require(self.candidate_id == CANDIDATE_C and self.tap_name == TAP_C, "quantized RDHA candidate/tap binding drift")
        _require(self.U.shape == (JOINT_PROJ_INPUT_DIM, RANK) and self.U.group_axis == "column", "quantized RDHA U layout drift")
        _require(self.V.shape == (RANK, JOINT_PROJ_INPUT_DIM) and self.V.group_axis == "row", "quantized RDHA V layout drift")
        _require(self.Q.shape == (RANK, SUMMARY_DIM) and self.Q.group_axis == "row", "quantized RDHA Q layout drift")
        _require(self.b.width == RANK, "quantized RDHA b width drift")
        _require(self.mean_p1.width == SUMMARY_DIM and self.std_p1.width == SUMMARY_DIM and self.a_max.width == 1, "quantized RDHA statistic width drift")
        self.std_p1.require_positive(name="quantized RDHA std")
        self.a_max.require_positive(name="quantized RDHA a_max")
        _require(self.numeric_payload_bytes == 1328, "quantized RDHA byte formula drift")

    @property
    def numeric_payload_bytes(self) -> int:
        return (
            self.U.nbytes
            + self.V.nbytes
            + self.Q.nbytes
            + self.b.nbytes
            + self.mean_p1.nbytes
            + self.std_p1.nbytes
            + self.a_max.nbytes
        )

    @property
    def persistent_fp32_sidecar(self) -> bool:
        return False

    def decode(self, *, device: torch.device | str = "cpu") -> RDHAAsset:
        return RDHAAsset(
            U=self.U.decode(device=device),
            V=self.V.decode(device=device),
            Q=self.Q.decode(device=device),
            b=self.b.decode(device=device),
            mean_p1=self.mean_p1.decode(device=device),
            std_p1=self.std_p1.decode(device=device),
            a_max=float(self.a_max.decode(device=device)[0].item()),
            candidate_id=CANDIDATE_C,
            tap_name=TAP_C,
        )


def quantize_fsrg_asset(asset: FSRGAsset) -> QuantizedFSRGAsset:
    """Encode a finalized float A/B asset in its only permitted state layout."""

    if not isinstance(asset, FSRGAsset):
        raise D127Phase1AssetError("quantize_fsrg_asset requires an FSRGAsset")
    return QuantizedFSRGAsset(
        candidate_id=asset.candidate_id,
        tap_name=asset.tap_name,
        U=SymmetricInt8Matrix.from_tensor(asset.U, group_axis="column", name="FSRG U"),
        V=SymmetricInt8Matrix.from_tensor(asset.V, group_axis="row", name="FSRG V"),
        d_f_diag=FP16Buffer.from_tensor(
            asset.d_f_diag, name="FSRG D_F", require_positive=True
        ),
        rho=FP16Buffer.from_tensor(
            torch.tensor([asset.rho], dtype=torch.float32),
            name="FSRG rho",
            require_positive=True,
        ),
    )


def quantize_rdah_asset(asset: RDHAAsset) -> QuantizedRDHAAsset:
    """Encode a finalized float C asset in its only permitted 1328-byte layout."""

    if not isinstance(asset, RDHAAsset):
        raise D127Phase1AssetError("quantize_rdah_asset requires an RDHAAsset")
    return QuantizedRDHAAsset(
        U=SymmetricInt8Matrix.from_tensor(asset.U, group_axis="column", name="RDHA U"),
        V=SymmetricInt8Matrix.from_tensor(asset.V, group_axis="row", name="RDHA V"),
        Q=SymmetricInt8Matrix.from_tensor(asset.Q, group_axis="row", name="RDHA Q"),
        b=SymmetricInt8Vector.from_tensor(asset.b, name="RDHA b"),
        mean_p1=FP16Buffer.from_tensor(asset.mean_p1, name="RDHA mean_p1"),
        std_p1=FP16Buffer.from_tensor(
            asset.std_p1, name="RDHA std_p1", require_positive=True
        ),
        a_max=FP16Buffer.from_tensor(
            torch.tensor([asset.a_max], dtype=torch.float32),
            name="RDHA a_max",
            require_positive=True,
        ),
    )


def assert_no_persistent_fp32_sidecar(
    asset: QuantizedFSRGAsset | QuantizedRDHAAsset,
) -> None:
    """Fail if a typed persistent asset accidentally carries tensor/array state."""

    if not isinstance(asset, (QuantizedFSRGAsset, QuantizedRDHAAsset)):
        raise D127Phase1AssetError("sidecar audit accepts only typed quantized D127 assets")

    def walk(value: object) -> None:
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            raise D127Phase1AssetError("quantized persistent asset carries a forbidden tensor sidecar")
        if hasattr(value, "__dataclass_fields__"):
            for item in fields(value):
                walk(getattr(value, item.name))
        elif isinstance(value, tuple):
            for item in value:
                walk(item)

    walk(asset)


@dataclass(frozen=True, slots=True)
class FunctionArgmaxParityReceipt:
    """A non-target Phase1 fixture receipt for quantization audit only."""

    fixture_id: str
    output_shape: tuple[int, ...]
    element_count: int
    max_abs_error: float
    mean_abs_error: float
    argmax_agreement: float
    argmax_equal: bool

    def __post_init__(self) -> None:
        _require(bool(self.fixture_id), "parity fixture_id must be nonempty")
        _require(len(self.output_shape) >= 2 and self.output_shape[-1] >= 2, "parity outputs require a nontrivial final argmax axis")
        _require(self.element_count >= 1, "parity receipt must contain outputs")
        _require(
            all(math.isfinite(float(value)) for value in (self.max_abs_error, self.mean_abs_error, self.argmax_agreement)),
            "parity receipt contains non-finite values",
        )
        _require(self.max_abs_error >= 0.0 and self.mean_abs_error >= 0.0, "parity errors must be nonnegative")
        _require(0.0 <= self.argmax_agreement <= 1.0, "argmax agreement must be a fraction")


def function_argmax_parity_receipt(
    *, fixture_id: str, reference_output: Tensor, quantized_output: Tensor
) -> FunctionArgmaxParityReceipt:
    """Summarize function and argmax parity without inventing a score threshold."""

    for name, value in (("reference_output", reference_output), ("quantized_output", quantized_output)):
        if (
            not torch.is_tensor(value)
            or not value.dtype.is_floating_point
            or value.ndim < 2
            or int(value.shape[-1]) < 2
            or not bool(torch.isfinite(value).all().item())
        ):
            raise D127Phase1AssetError(f"{name} must be finite floating [...,class>=2]")
    if tuple(reference_output.shape) != tuple(quantized_output.shape):
        raise D127Phase1AssetError("parity output shape drift")
    reference = reference_output.detach().to(device="cpu", dtype=torch.float64)
    quantized = quantized_output.detach().to(device="cpu", dtype=torch.float64)
    absolute = torch.abs(reference - quantized)
    reference_argmax = torch.argmax(reference, dim=-1)
    quantized_argmax = torch.argmax(quantized, dim=-1)
    agreement = torch.eq(reference_argmax, quantized_argmax)
    return FunctionArgmaxParityReceipt(
        fixture_id=fixture_id,
        output_shape=tuple(int(item) for item in reference.shape),
        element_count=int(reference.numel()),
        max_abs_error=float(absolute.max().item()),
        mean_abs_error=float(absolute.mean().item()),
        argmax_agreement=float(agreement.to(dtype=torch.float64).mean().item()),
        argmax_equal=bool(torch.all(agreement).item()),
    )


Phase1FixtureForward = Callable[[FSRGAsset | RDHAAsset], Tensor]


def phase1_fixture_parity_receipt(
    *,
    fixture_id: str,
    float_asset: FSRGAsset | RDHAAsset,
    quantized_asset: QuantizedFSRGAsset | QuantizedRDHAAsset,
    forward: Phase1FixtureForward,
) -> FunctionArgmaxParityReceipt:
    """Run a caller-provided Phase1 fixture before discarding its FP32 asset."""

    if not callable(forward):
        raise D127Phase1AssetError("Phase1 fixture forward must be callable")
    if isinstance(float_asset, FSRGAsset) and not isinstance(quantized_asset, QuantizedFSRGAsset):
        raise D127Phase1AssetError("FSRG fixture requires a matching quantized FSRG asset")
    if isinstance(float_asset, RDHAAsset) and not isinstance(quantized_asset, QuantizedRDHAAsset):
        raise D127Phase1AssetError("RDHA fixture requires a matching quantized RDHA asset")
    if not isinstance(float_asset, (FSRGAsset, RDHAAsset)):
        raise D127Phase1AssetError("fixture requires a frozen D127 float asset")
    assert_no_persistent_fp32_sidecar(quantized_asset)
    with torch.no_grad():
        reference = forward(float_asset)
        decoded = quantized_asset.decode()
        quantized = forward(decoded)
    return function_argmax_parity_receipt(
        fixture_id=fixture_id,
        reference_output=reference,
        quantized_output=quantized,
    )


__all__ = [
    "CanonicalRank2Initialization",
    "DeterministicLBFGSReceipt",
    "D127Phase1AssetError",
    "FP16Buffer",
    "FSRGEpisode",
    "FSRGLossCallbacks",
    "FSRGPhase1TrainingResult",
    "FSRGSealedStatistics",
    "FrozenFSRGEpisodes",
    "FrozenRDHAEpisodes",
    "FunctionArgmaxParityReceipt",
    "LBFGS_LINE_SEARCH",
    "LBFGS_MAX_ITER",
    "Phase1FixtureForward",
    "QuantizedFSRGAsset",
    "QuantizedRDHAAsset",
    "RDHAEpisode",
    "RDHALossCallback",
    "RDHAPhase1TrainingResult",
    "RDHASealedStatistics",
    "SymmetricInt8Matrix",
    "SymmetricInt8Vector",
    "assert_no_persistent_fp32_sidecar",
    "canonical_receiver_mean_svd",
    "function_argmax_parity_receipt",
    "phase1_fixture_parity_receipt",
    "quantize_fsrg_asset",
    "quantize_rdah_asset",
    "seal_fsrg_statistics",
    "seal_rdah_statistics",
    "train_fsrg_phase1_asset",
    "train_rdah_phase1_asset",
]
